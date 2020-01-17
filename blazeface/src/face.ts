/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box, createBox, disposeBox, scaleBox} from './box';

/*
 * The object describing a face.
 */
export interface NormalizedFace {
  /** The upper left-hand corner of the face. */
  topLeft: [number, number]|tf.Tensor1D;
  /** The lower right-hand corner of the face. */
  bottomRight: [number, number]|tf.Tensor1D;
  /** Facial landmark coordinates. */
  landmarks: number[][]|tf.Tensor2D;
  /** Probability of the face detection. */
  probability: number|tf.Tensor1D;
}

// The blazeface model predictions containing unnormalized coordinates
// for facial bounding box / landmarks.
type BlazeFacePrediction = {
  box: Box,
  landmarks: tf.Tensor2D,
  probability: tf.Tensor1D,
  anchor: tf.Tensor2D|[number, number]
};

// Blazeface scatters anchor points throughout the input image and for each
// point predicts the probability that it lies within a face. `ANCHORS_CONFIG`
// is a fixed configuration that determines where the anchor points are
// scattered.
const ANCHORS_CONFIG = {
  'strides': [8, 16],
  'anchors': [2, 6]
};

// `NUM_LANDMARKS` is a fixed property of the model.
const NUM_LANDMARKS = 6;

function generateAnchors(
    width: number, height: number,
    outputSpec: {strides: [number, number], anchors: [number, number]}):
    number[][] {
  const anchors = [];
  for (let i = 0; i < outputSpec.strides.length; i++) {
    const stride = outputSpec.strides[i];
    const gridRows = Math.floor((height + stride - 1) / stride);
    const gridCols = Math.floor((width + stride - 1) / stride);
    const anchorsNum = outputSpec.anchors[i];

    for (let gridY = 0; gridY < gridRows; gridY++) {
      const anchorY = stride * (gridY + 0.5);

      for (let gridX = 0; gridX < gridCols; gridX++) {
        const anchorX = stride * (gridX + 0.5);
        for (let n = 0; n < anchorsNum; n++) {
          anchors.push([anchorX, anchorY]);
        }
      }
    }
  }

  return anchors;
}

function decodeBounds(
    boxOutputs: tf.Tensor2D, anchors: tf.Tensor2D,
    inputSize: tf.Tensor1D): tf.Tensor2D {
  const boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
  const centers = tf.add(boxStarts, anchors);
  const boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);

  const boxSizesNormalized = tf.div(boxSizes, inputSize);
  const centersNormalized = tf.div(centers, inputSize);

  const halfBoxSize = tf.div(boxSizesNormalized, 2);
  const starts = tf.sub(centersNormalized, halfBoxSize);
  const ends = tf.add(centersNormalized, halfBoxSize);

  const startNormalized = tf.mul(starts, inputSize);
  const endNormalized = tf.mul(ends, inputSize);

  const concatAxis = 1;
  return tf.concat2d(
      [startNormalized as tf.Tensor2D, endNormalized as tf.Tensor2D],
      concatAxis);
}

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipFaceHorizontal(
    face: NormalizedFace, imageWidth: number): NormalizedFace {
  if (face.topLeft instanceof tf.Tensor) {
    return {
      topLeft: tf.concat([
        tf.sub(imageWidth - 1, face.topLeft.slice(0, 1)),
        face.topLeft.slice(1, 1)
      ]) as tf.Tensor1D,
      bottomRight: tf.concat([
        tf.sub(imageWidth - 1, (face.bottomRight as tf.Tensor).slice(0, 1)),
        (face.bottomRight as tf.Tensor).slice(1, 1)
      ]) as tf.Tensor1D,
      landmarks: tf.sub(tf.tensor1d([imageWidth - 1, 0]), face.landmarks)
                     .mul(tf.tensor1d([1, -1])) as tf.Tensor2D,
      probability: face.probability
    } as NormalizedFace;
  }

  return {
    topLeft: [imageWidth - 1 - face.topLeft[0], face.topLeft[1]],
    bottomRight: [
      imageWidth - 1 - (face.bottomRight as [number, number])[0],
      (face.bottomRight as [number, number])[1]
    ],
    landmarks:
        (face.landmarks as number[][]).map((coord: [number, number]) => ([
                                             imageWidth - 1 - coord[0], coord[1]
                                           ])),
    probability: face.probability
  };
}

export class BlazeFaceModel {
  private blazeFaceModel: tfconv.GraphModel;
  private width: number;
  private height: number;
  private maxFaces: number;
  private anchors: tf.Tensor2D;
  private anchorsData: number[][];
  private inputSize: tf.Tensor1D;
  private inputSizeData: [number, number];
  private iouThreshold: number;
  private scoreThreshold: number;

  constructor(
      model: tfconv.GraphModel, width: number, height: number, maxFaces: number,
      iouThreshold: number, scoreThreshold: number) {
    this.blazeFaceModel = model;
    this.width = width;
    this.height = height;
    this.maxFaces = maxFaces;
    this.anchorsData = generateAnchors(
        width, height,
        ANCHORS_CONFIG as
            {strides: [number, number], anchors: [number, number]});
    this.anchors = tf.tensor2d(this.anchorsData);
    this.inputSizeData = [width, height];
    this.inputSize = tf.tensor1d([width, height]);

    this.iouThreshold = iouThreshold;
    this.scoreThreshold = scoreThreshold;
  }

  async getBoundingBoxes(inputImage: tf.Tensor4D, returnTensors: boolean):
      Promise<[BlazeFacePrediction[], tf.Tensor|[number, number]]> {
    const [detectedOutputs, boxes, scores] = tf.tidy(() => {
      const resizedImage = inputImage.resizeBilinear([this.width, this.height]);
      const normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);

      // [1, 897, 17] 1 = batch, 897 = number of anchors
      const batchedPrediction = this.blazeFaceModel.predict(normalizedImage);
      const prediction = (batchedPrediction as tf.Tensor3D).squeeze();

      const decodedBounds =
          decodeBounds(prediction as tf.Tensor2D, this.anchors, this.inputSize);
      const logits = tf.slice(prediction as tf.Tensor2D, [0, 0], [-1, 1]);
      return [prediction, decodedBounds, tf.sigmoid(logits).squeeze()];
    });

    // TODO: Once tf.image.nonMaxSuppression includes a flag to suppress console
    // warnings for not using async version, pass that flag in.
    const savedConsoleWarnFn = console.warn;
    console.warn = () => {};

    const boxIndicesTensor = tf.image.nonMaxSuppression(
        boxes as tf.Tensor2D, scores as tf.Tensor1D, this.maxFaces,
        this.iouThreshold, this.scoreThreshold);
    console.warn = savedConsoleWarnFn;
    const boxIndices = await boxIndicesTensor.array();
    boxIndicesTensor.dispose();

    let boundingBoxes: tf.Tensor[]|number[][][] = boxIndices.map(
        (boxIndex: number) => tf.slice(boxes, [boxIndex, 0], [1, -1]));
    if (!returnTensors) {
      boundingBoxes = await Promise.all(
          boundingBoxes.map(async (boundingBox: tf.Tensor2D) => {
            const vals = await boundingBox.array();
            boundingBox.dispose();
            return vals;
          }));
    }

    const originalHeight = inputImage.shape[1];
    const originalWidth = inputImage.shape[2];

    let scaleFactor: tf.Tensor|[number, number];
    if (returnTensors) {
      scaleFactor = tf.div([originalWidth, originalHeight], this.inputSize);
    } else {
      scaleFactor = [
        originalWidth / this.inputSizeData[0],
        originalHeight / this.inputSizeData[1]
      ];
    }

    const annotatedBoxes =
        (boundingBoxes as number[][][])
            .map(
                (boundingBox: tf.Tensor2D|number[][], i: number) =>
                    tf.tidy(() => {
                      const boxIndex = boxIndices[i];

                      let anchor;
                      if (returnTensors) {
                        anchor = this.anchors.slice([boxIndex, 0], [1, 2]);
                      } else {
                        anchor = this.anchorsData[boxIndex] as [number, number];
                      }

                      const box = boundingBox instanceof tf.Tensor ?
                          createBox(boundingBox) :
                          createBox(tf.tensor2d(boundingBox));
                      const landmarks =
                          tf.slice(
                                detectedOutputs, [boxIndex, NUM_LANDMARKS - 1],
                                [1, -1])
                              .squeeze()
                              .reshape([NUM_LANDMARKS, -1]);
                      const probability = tf.slice(scores, [boxIndex], [1]);

                      return {box, landmarks, probability, anchor};
                    }));

    boxes.dispose();
    scores.dispose();
    detectedOutputs.dispose();

    return [annotatedBoxes as BlazeFacePrediction[], scaleFactor];
  }

  /**
   * Returns an array of faces in an image.
   *
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param returnTensors (defaults to `false`) Whether to return tensors as
   * opposed to values.
   * @param flipHorizontal Whether to flip/mirror the facial keypoints
   * horizontally. Should be true for videos that are flipped by default (e.g.
   * webcams).
   *
   * @return An array of detected faces, each with the following properties:
   *  `topLeft`: the upper left coordinate of the face in the form `[x, y]`
   *  `bottomRight`: the lower right coordinate of the face in the form `[x, y]`
   *  `landmarks`: facial landmark coordinates
   *  `probability`: the probability of the face being present
   */
  async estimateFaces(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false,
      flipHorizontal = false): Promise<NormalizedFace[]> {
    const [, width] = getInputTensorDimensions(input);
    const image = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });
    const [prediction, scaleFactor] =
        await this.getBoundingBoxes(image as tf.Tensor4D, returnTensors);
    image.dispose();

    if (returnTensors) {
      return prediction.map((face: BlazeFacePrediction) => {
        const scaledBox = scaleBox(face.box, scaleFactor as tf.Tensor1D)
                              .startEndTensor.squeeze();

        let normalizedFace = {
          topLeft: scaledBox.slice([0], [2]),
          bottomRight: scaledBox.slice([2], [2]),
          landmarks: face.landmarks.add(face.anchor).mul(scaleFactor),
          probability: face.probability
        } as NormalizedFace;
        if (flipHorizontal) {
          normalizedFace =
              flipFaceHorizontal(normalizedFace as NormalizedFace, width) as
              NormalizedFace;
        }
        return normalizedFace;
      });
    }

    return Promise.all(prediction.map(async (face: BlazeFacePrediction) => {
      const scaledBox = tf.tidy(() => {
        return scaleBox(face.box, scaleFactor as [number, number])
            .startEndTensor.squeeze();
      });

      const [landmarkData, boxData, probabilityData] =
          await Promise.all([face.landmarks, scaledBox, face.probability].map(
              async d => d.array()));

      const anchor = face.anchor as [number, number];
      const scaledLandmarks =
          (landmarkData as number[][])
              .map((landmark: [number, number]) => ([
                     (landmark[0] + anchor[0]) *
                         (scaleFactor as [number, number])[0],
                     (landmark[1] + anchor[1]) *
                         (scaleFactor as [number, number])[1]
                   ]));

      scaledBox.dispose();
      disposeBox(face.box);
      face.landmarks.dispose();
      face.probability.dispose();

      let normalizedFace = {
        topLeft: (boxData as number[]).slice(0, 2),
        bottomRight: (boxData as number[]).slice(2),
        landmarks: scaledLandmarks,
        probability: probabilityData
      } as NormalizedFace;

      if (flipHorizontal) {
        normalizedFace = flipFaceHorizontal(normalizedFace, width);
      }

      return normalizedFace;
    }));
  }
}
