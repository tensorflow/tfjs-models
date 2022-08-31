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
  landmarks?: number[][]|tf.Tensor2D;
  /** Probability of the face detection. */
  probability?: number|tf.Tensor1D;
}

// The blazeface model predictions containing unnormalized coordinates
// for facial bounding box / landmarks.
export type BlazeFacePrediction = {
  box: Box,
  landmarks: tf.Tensor2D,
  probability: tf.Tensor1D,
  anchor: tf.Tensor2D|[number, number]
};

// Blazeface scatters anchor points throughout the input image and for each
// point predicts the probability that it lies within a face. `ANCHORS_CONFIG`
// is a fixed configuration that determines where the anchor points are
// scattered.
declare interface AnchorsConfig {
  strides: [number, number];
  anchors: [number, number];
}
const ANCHORS_CONFIG: AnchorsConfig = {
  'strides': [8, 16],
  'anchors': [2, 6]
};

// `NUM_LANDMARKS` is a fixed property of the model.
const NUM_LANDMARKS = 6;

function generateAnchors(
    width: number, height: number, outputSpec: AnchorsConfig): number[][] {
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
  let flippedTopLeft: [number, number]|tf.Tensor1D,
      flippedBottomRight: [number, number]|tf.Tensor1D,
      flippedLandmarks: number[][]|tf.Tensor2D;

  if (face.topLeft instanceof tf.Tensor &&
      face.bottomRight instanceof tf.Tensor) {
    const [topLeft, bottomRight] = tf.tidy(() => {
      return [
        tf.concat([
          tf.slice(tf.sub(imageWidth - 1, (face.topLeft as tf.Tensor)), 0, 1),
          tf.slice((face.topLeft as tf.Tensor), 1, 1)
        ]) as tf.Tensor1D,
        tf.concat([
          tf.sub(imageWidth - 1,
            tf.slice((face.bottomRight as tf.Tensor), 0, 1)),
          tf.slice((face.bottomRight as tf.Tensor), 1, 1)
        ]) as tf.Tensor1D
      ];
    });

    flippedTopLeft = topLeft;
    flippedBottomRight = bottomRight;

    if (face.landmarks != null) {
      flippedLandmarks = tf.tidy(() => {
        const a: tf.Tensor2D =
            tf.sub(tf.tensor1d([imageWidth - 1, 0]), face.landmarks);
        const b = tf.tensor1d([1, -1]);
        const product: tf.Tensor2D = tf.mul(a, b);
        return product;
      });
    }
  } else {
    const [topLeftX, topLeftY] = face.topLeft as [number, number];
    const [bottomRightX, bottomRightY] = face.bottomRight as [number, number];

    flippedTopLeft = [imageWidth - 1 - topLeftX, topLeftY];
    flippedBottomRight = [imageWidth - 1 - bottomRightX, bottomRightY];

    if (face.landmarks != null) {
      flippedLandmarks =
          (face.landmarks as number[][]).map((coord: [number, number]) => ([
                                               imageWidth - 1 - coord[0],
                                               coord[1]
                                             ]));
    }
  }

  const flippedFace: NormalizedFace = {
    topLeft: flippedTopLeft,
    bottomRight: flippedBottomRight
  };

  if (flippedLandmarks != null) {
    flippedFace.landmarks = flippedLandmarks;
  }

  if (face.probability != null) {
    flippedFace.probability = face.probability instanceof tf.Tensor ?
        face.probability.clone() :
        face.probability;
  }

  return flippedFace;
}

function scaleBoxFromPrediction(
    face: BlazeFacePrediction|Box, scaleFactor: tf.Tensor1D|[number, number]) {
  return tf.tidy(() => {
    let box;
    if (face.hasOwnProperty('box')) {
      box = (face as BlazeFacePrediction).box;
    } else {
      box = face;
    }
    return tf.squeeze(scaleBox(box as Box, scaleFactor).startEndTensor);
  });
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

  async getBoundingBoxes(
      inputImage: tf.Tensor4D, returnTensors: boolean,
      annotateBoxes = true): Promise<{
    boxes: Array<BlazeFacePrediction|Box>,
    scaleFactor: tf.Tensor|[number, number]
  }> {
    const [detectedOutputs, boxes, scores] = tf.tidy((): [
      tf.Tensor2D, tf.Tensor2D, tf.Tensor1D
    ] => {
      const resizedImage = tf.image.resizeBilinear(inputImage,
        [this.width, this.height]);
      const normalizedImage = tf.mul(tf.sub(tf.div(resizedImage, 255), 0.5), 2);

      // [1, 897, 17] 1 = batch, 897 = number of anchors
      const batchedPrediction = this.blazeFaceModel.predict(normalizedImage);
      const prediction = tf.squeeze((batchedPrediction as tf.Tensor3D));

      const decodedBounds =
          decodeBounds(prediction as tf.Tensor2D, this.anchors, this.inputSize);
      const logits = tf.slice(prediction as tf.Tensor2D, [0, 0], [-1, 1]);
      const scores = tf.squeeze(tf.sigmoid(logits));
      return [prediction as tf.Tensor2D, decodedBounds, scores as tf.Tensor1D];
    });

    // TODO: Once tf.image.nonMaxSuppression includes a flag to suppress console
    // warnings for not using async version, pass that flag in.
    const savedConsoleWarnFn = console.warn;
    console.warn = () => {};
    const boxIndicesTensor = tf.image.nonMaxSuppression(
        boxes, scores, this.maxFaces, this.iouThreshold, this.scoreThreshold);
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

    const annotatedBoxes = [];
    for (let i = 0; i < boundingBoxes.length; i++) {
      const boundingBox = boundingBoxes[i] as tf.Tensor2D | number[][];
      const annotatedBox = tf.tidy(() => {
        const box = boundingBox instanceof tf.Tensor ?
            createBox(boundingBox) :
            createBox(tf.tensor2d(boundingBox));

        if (!annotateBoxes) {
          return box;
        }

        const boxIndex = boxIndices[i];

        let anchor;
        if (returnTensors) {
          anchor = tf.slice(this.anchors, [boxIndex, 0], [1, 2]);
        } else {
          anchor = this.anchorsData[boxIndex] as [number, number];
        }

        const landmarks = tf.reshape(tf.squeeze(tf.slice(detectedOutputs,
          [boxIndex, NUM_LANDMARKS - 1], [1, -1])), [NUM_LANDMARKS, -1]);
        const probability = tf.slice(scores, [boxIndex], [1]);

        return {box, landmarks, probability, anchor};
      });
      annotatedBoxes.push(annotatedBox);
    }

    boxes.dispose();
    scores.dispose();
    detectedOutputs.dispose();

    return {
      boxes: annotatedBoxes as Array<BlazeFacePrediction|Box>,
      scaleFactor
    };
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
   * @param annotateBoxes (defaults to `true`) Whether to annotate bounding
   * boxes with additional properties such as landmarks and probability. Pass in
   * `false` for faster inference if annotations are not needed.
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
      returnTensors = false, flipHorizontal = false,
      annotateBoxes = true): Promise<NormalizedFace[]> {
    const [, width] = getInputTensorDimensions(input);
    const image = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast((input as tf.Tensor), 'float32'), 0);
    });
    const {boxes, scaleFactor} = await this.getBoundingBoxes(
        image as tf.Tensor4D, returnTensors, annotateBoxes);
    image.dispose();

    if (returnTensors) {
      return boxes.map((face: BlazeFacePrediction|Box) => {
        const scaledBox =
            scaleBoxFromPrediction(face, scaleFactor as tf.Tensor1D);
        let normalizedFace: NormalizedFace = {
          topLeft: tf.slice(scaledBox, [0], [2]) as tf.Tensor1D,
          bottomRight: tf.slice(scaledBox, [2], [2]) as tf.Tensor1D
        };

        if (annotateBoxes) {
          const {landmarks, probability, anchor} = face as {
            landmarks: tf.Tensor2D,
            probability: tf.Tensor1D,
            anchor: tf.Tensor2D | [number, number]
          };

          const normalizedLandmarks: tf.Tensor2D =
              tf.mul(tf.add(landmarks, anchor), scaleFactor);
          normalizedFace.landmarks = normalizedLandmarks;
          normalizedFace.probability = probability;
        }

        if (flipHorizontal) {
          normalizedFace = flipFaceHorizontal(normalizedFace, width);
        }
        return normalizedFace;
      });
    }

    return Promise.all(boxes.map(async (face: BlazeFacePrediction) => {
      const scaledBox =
          scaleBoxFromPrediction(face, scaleFactor as [number, number]);
      let normalizedFace: NormalizedFace;
      if (!annotateBoxes) {
        const boxData = await scaledBox.array();
        normalizedFace = {
          topLeft: (boxData as number[]).slice(0, 2) as [number, number],
          bottomRight: (boxData as number[]).slice(2) as [number, number]
        };
      } else {
        const [landmarkData, boxData, probabilityData] =
            await Promise.all([face.landmarks, scaledBox, face.probability].map(
                async d => d.array()));

        const anchor = face.anchor as [number, number];
        const [scaleFactorX, scaleFactorY] = scaleFactor as [number, number];
        const scaledLandmarks =
            (landmarkData as Array<[number, number]>)
                .map(landmark => ([
                       (landmark[0] + anchor[0]) * scaleFactorX,
                       (landmark[1] + anchor[1]) * scaleFactorY
                     ]));

        normalizedFace = {
          topLeft: (boxData as number[]).slice(0, 2) as [number, number],
          bottomRight: (boxData as number[]).slice(2) as [number, number],
          landmarks: scaledLandmarks,
          probability: probabilityData as number
        };

        disposeBox(face.box);
        face.landmarks.dispose();
        face.probability.dispose();
      }

      scaledBox.dispose();

      if (flipHorizontal) {
        normalizedFace = flipFaceHorizontal(normalizedFace, width);
      }

      return normalizedFace;
    }));
  }

  /**
   * Dispose the WebGL memory held by the underlying model.
   */
  dispose(): void {
    if (this.blazeFaceModel != null) {
      this.blazeFaceModel.dispose();
    }
  }
}
