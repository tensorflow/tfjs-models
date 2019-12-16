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

import {createBox, disposeBox, scaleBox} from './box';

const ANCHORS_CONFIG = {
  'strides': [8, 16],
  'anchors': [2, 6]
};

const generateAnchors =
    (width: number, height: number,
     outputSpec: {strides: [number, number], anchors: [number, number]}):
        number[][] => {
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
        };

const decodeBounds = (boxOutputs: tf.Tensor2D, anchors: tf.Tensor2D,
                      inputSize: tf.Tensor1D): tf.Tensor2D => {
  const boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
  const centers = tf.add(boxStarts, anchors);

  const boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);

  const boxSizesNormalized = tf.div(boxSizes, inputSize);
  const centersNormalized = tf.div(centers, inputSize);

  const starts = tf.sub(centersNormalized, tf.div(boxSizesNormalized, 2));
  const ends = tf.add(centersNormalized, tf.div(boxSizesNormalized, 2));

  return tf.concat2d(
      [
        tf.mul(starts, inputSize) as tf.Tensor2D,
        tf.mul(ends, inputSize) as tf.Tensor2D
      ],
      1);
};

/**
 * Feeds the input to the model and decodes the facial bounding boxes (if any)
 * from the model output.
 */
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

  async getBoundingBoxes(inputImage: tf.Tensor4D, returnTensors: boolean) {
    const [detectedOutputs, boxes, scores] = tf.tidy(() => {
      const resizedImage = inputImage.resizeBilinear([this.width, this.height]);
      const normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);

      const prediction =
          (this.blazeFaceModel.predict(normalizedImage) as tf.Tensor3D)
              .squeeze() as tf.Tensor2D;

      const decodedBounds =
          decodeBounds(prediction, this.anchors, this.inputSize);
      const logits = tf.slice(prediction as tf.Tensor2D, [0, 0], [-1, 1]);
      return [prediction, decodedBounds, tf.sigmoid(logits).squeeze()];
    });

    const boxIndicesTensor = tf.image.nonMaxSuppression(
        boxes as tf.Tensor2D, scores as tf.Tensor1D, this.maxFaces,
        this.iouThreshold, this.scoreThreshold);
    const boxIndices = await boxIndicesTensor.array();
    boxIndicesTensor.dispose();

    const boundingBoxes =
        await Promise.all(boxIndices.map(async (boxIndex: number) => {
          const box = tf.slice(boxes, [boxIndex, 0], [1, -1]);
          const vals = await box.array();
          box.dispose();
          return vals;
        }));

    const originalHeight = inputImage.shape[1];
    const originalWidth = inputImage.shape[2];

    let scaleFactor: [number, number]|tf.Tensor1D;
    if (returnTensors) {
      scaleFactor = tf.div([originalWidth, originalHeight], this.inputSize) as
          tf.Tensor1D;
    } else {
      scaleFactor = [
        originalWidth / this.inputSizeData[0],
        originalHeight / this.inputSizeData[1]
      ];
    }

    const annotatedBoxes = boundingBoxes.map((boundingBox, i) => tf.tidy(() => {
      const boxIndex = boxIndices[i];

      let anchor: [number, number]|tf.Tensor2D;
      if (returnTensors) {
        anchor = this.anchors.slice([boxIndex, 0], [1, 2]) as tf.Tensor2D;
      } else {
        anchor = this.anchorsData[boxIndex] as [number, number];
      }

      const box = createBox(tf.tensor2d(boundingBox as number[][]));
      const landmarks =
          tf.slice(detectedOutputs, [boxIndex, 5], [1, -1]).squeeze().reshape([
            6, -1
          ]);
      const probability = tf.slice(scores, [boxIndex], [1]);

      return {box, landmarks, probability, anchor};
    }));

    boxes.dispose();
    scores.dispose();
    detectedOutputs.dispose();

    return [annotatedBoxes, scaleFactor];
  }

  /**
   * Returns an array of faces in an image.
   *
   * @param input The image to classify. Can be a tensor or a DOM element iamge,
   * video, or canvas.
   */
  async estimateFace(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false): Promise<any> {
    const image = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0) as tf.Tensor4D;
    });
    const [prediction, scaleFactor] =
        await this.getBoundingBoxes(image as tf.Tensor4D, returnTensors);
    image.dispose();

    if (returnTensors) {
      return (prediction as any[]).map((d: any) => {
        const scaledBox = scaleBox(d.box, scaleFactor as tf.Tensor1D)
                              .startEndTensor.squeeze();

        return {
          topLeft: scaledBox.slice([0], [2]),
          bottomRight: scaledBox.slice([2], [2]),
          landmarks: d.landmarks.add(d.anchor).mul(scaleFactor),
          probability: d.probability
        };
      });
    }

    return Promise.all((prediction as any[]).map(async (d: any) => {
      const scaledBox = tf.tidy(() => {
        return scaleBox(d.box, scaleFactor as [number, number])
            .startEndTensor.squeeze();
      });

      const [landmarkData, boxData, probabilityData] =
          await Promise.all([d.landmarks, scaledBox, d.probability].map(
              async innerD => innerD.array()));

      const anchor = d.anchor as [number, number];
      const scaledLandmarks = landmarkData.map(
          (landmark: [number, number]) => ([
            (landmark[0] + anchor[0]) * (scaleFactor as [number, number])[0],
            (landmark[1] + anchor[1]) * (scaleFactor as [number, number])[1]
          ]));

      scaledBox.dispose();
      disposeBox(d.box);
      d.landmarks.dispose();
      d.probability.dispose();

      return {
        topLeft: (boxData as number[]).slice(0, 2),
        bottomRight: (boxData as number[]).slice(2),
        landmarks: scaledLandmarks,
        probability: probabilityData
      };
    }));
  }
}
