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

import {Box} from './box';

type AnchorsConfig = {
  strides: [number, number],
  anchors: [number, number]
};

export class BlazeFaceModel {
  private blazeFaceModel: tfconv.GraphModel;
  private width: number;
  private height: number;
  private config: AnchorsConfig;
  private anchors: tf.Tensor;
  private inputSize: tf.Tensor;
  private iouThreshold: number;
  private scoreThreshold: number;

  constructor(model: tfconv.GraphModel, width: number, height: number) {
    this.blazeFaceModel = model;
    this.width = width;
    this.height = height;
    this.config = this.getAnchorsConfig();
    this.anchors = this.generateAnchors(width, height, this.config);
    this.inputSize = tf.tensor([width, height]);

    this.iouThreshold = 0.3;
    this.scoreThreshold = 0.75;
  }

  getAnchorsConfig(): AnchorsConfig {
    return {
      'strides': [8, 16],
      'anchors': [2, 6],
    };
  }

  generateAnchors(width: number, height: number, outputSpec: AnchorsConfig):
      tf.Tensor {
    const anchors = [];
    for (let i = 0; i < outputSpec.strides.length; ++i) {
      const stride = outputSpec.strides[i];
      const gridRows = Math.floor((height + stride - 1) / stride);
      const gridCols = Math.floor((width + stride - 1) / stride);
      const anchorsNum = outputSpec.anchors[i];

      for (let gridY = 0; gridY < gridRows; ++gridY) {
        const anchorY = stride * (gridY + 0.5);

        for (let gridX = 0; gridX < gridCols; ++gridX) {
          const anchorX = stride * (gridX + 0.5);
          for (let n = 0; n < anchorsNum; n++) {
            anchors.push([anchorX, anchorY]);
          }
        }
      }
    }

    return tf.tensor(anchors);
  }

  decodeBounds(boxOutputs: tf.Tensor2D): tf.Tensor2D {
    const boxStarts = tf.slice(boxOutputs, [0, 0], [-1, 2]);
    const centers = tf.add(boxStarts, this.anchors);

    const boxSizes = tf.slice(boxOutputs, [0, 2], [-1, 2]);

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

  getBoundingBox(inputImage: tf.Tensor4D): number[][] {
    const normalizedImage = tf.mul(tf.sub(inputImage, 0.5), 2);
    const detectOutputs = this.blazeFaceModel.predict(normalizedImage);

    const scores =
        tf.sigmoid(
              tf.slice(detectOutputs as tf.Tensor3D, [0, 0, 0], [1, -1, 1]))
            .squeeze();

    const boxRegressors =
        tf.slice(detectOutputs as tf.Tensor3D, [0, 0, 1], [1, -1, 4]).squeeze();
    const boxes = this.decodeBounds(boxRegressors as tf.Tensor2D);
    const boxIndices = tf.image
                           .nonMaxSuppression(
                               boxes, scores as tf.Tensor1D, 1,
                               this.iouThreshold, this.scoreThreshold)
                           .arraySync();

    if (boxIndices.length === 0) {
      return null;  // TODO (vakunov): don't return null. Empty box?
    }

    // TODO (vakunov): change to multi face case
    const boxIndex = boxIndices[0];
    const resultBox = tf.slice(boxes, [boxIndex, 0], [1, -1]);
    return resultBox.arraySync();
  }

  getSingleBoundingBox(inputImage: tf.Tensor4D): Box {
    const originalHeight = inputImage.shape[1];
    const originalWidth = inputImage.shape[2];

    const image = inputImage.resizeBilinear([this.width, this.height]).div(255);
    const bboxes = this.getBoundingBox(image as tf.Tensor4D);

    if (!bboxes) {
      return null;
    }

    const factors = tf.div([originalWidth, originalHeight], this.inputSize);
    return new Box(tf.tensor(bboxes)).scale(factors as tf.Tensor1D);
  }
}
