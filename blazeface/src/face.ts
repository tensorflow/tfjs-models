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

import {createBox, scaleBox} from './box';

type AnchorsConfig = {
  strides: [number, number],
  anchors: [number, number]
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
  private config: AnchorsConfig;
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
    this.config = this.getAnchorsConfig();
    this.anchorsData = this.generateAnchors(width, height, this.config);
    this.anchors = tf.tensor2d(this.anchorsData);
    this.inputSizeData = [width, height];
    this.inputSize = tf.tensor1d([width, height]);

    this.iouThreshold = iouThreshold;
    this.scoreThreshold = scoreThreshold;
  }

  getAnchorsConfig(): AnchorsConfig {
    return {
      'strides': [8, 16],
      'anchors': [2, 6],
    };
  }

  generateAnchors(width: number, height: number, outputSpec: AnchorsConfig):
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

  decodeBounds(boxOutputs: tf.Tensor2D): tf.Tensor2D {
    const boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
    const centers = tf.add(boxStarts, this.anchors);

    const boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);

    const boxSizesNormalized = tf.div(boxSizes, this.inputSize);
    const centersNormalized = tf.div(centers, this.inputSize);

    const starts = tf.sub(centersNormalized, tf.div(boxSizesNormalized, 2));
    const ends = tf.add(centersNormalized, tf.div(boxSizesNormalized, 2));

    return tf.concat2d(
        [
          tf.mul(starts, this.inputSize) as tf.Tensor2D,
          tf.mul(ends, this.inputSize) as tf.Tensor2D
        ],
        1);
  }

  getBoundingBoxes(inputImage: tf.Tensor4D) {
    let normalizedImage =
        inputImage.resizeBilinear([this.width, this.height]).div(255);
    normalizedImage = tf.mul(tf.sub(normalizedImage, 0.5), 2);

    const detectedOutputs =
        (this.blazeFaceModel.predict(normalizedImage) as tf.Tensor3D)
            .squeeze() as tf.Tensor2D;

    const boxes = this.decodeBounds(detectedOutputs);
    const logits = tf.slice(detectedOutputs as tf.Tensor2D, [0, 0], [-1, 1]);
    const scores = tf.sigmoid(logits).squeeze();
    const boxIndices = tf.image
                           .nonMaxSuppression(
                               boxes, scores as tf.Tensor1D, this.maxFaces,
                               this.iouThreshold, this.scoreThreshold)
                           .arraySync();
    const boundingBoxes = boxIndices.map(boxIndex => {
      const landmarks = tf.slice(detectedOutputs, [boxIndex, 5], [1, -1])
                            .squeeze()
                            .reshape([6, -1])
                            .arraySync() as number[][];

      return {
        box: tf.slice(boxes, [boxIndex, 0], [1, -1]).arraySync(),
        probability: tf.slice(scores, [boxIndex], [1]).arraySync(),
        landmarks,
        anchor: this.anchorsData[boxIndex]
      };
    });

    const originalHeight = inputImage.shape[1];
    const originalWidth = inputImage.shape[2];
    const factors =
        tf.div([originalWidth, originalHeight], this.inputSize) as tf.Tensor1D;
    const factorsData = [
      originalWidth / this.inputSizeData[0],
      originalHeight / this.inputSizeData[1]
    ];

    return boundingBoxes.map(boundingBox => {
      const startEndTensor = tf.tensor2d(boundingBox.box);
      const box = createBox(startEndTensor);

      const scaledLandmarks = boundingBox.landmarks.map(
          landmark => landmark.map(
              (coord, coordIndex) => (coord + boundingBox.anchor[coordIndex]) *
                  factorsData[coordIndex]));

      return {
        box: scaleBox(box, factors).startEndTensor.squeeze(),
        landmarks: scaledLandmarks,
        probability: boundingBox.probability
      };
    });
  }
}
