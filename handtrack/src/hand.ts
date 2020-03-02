/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {scaleBoxCoordinates} from './box';

// function getInputTensorDimensions(input:
// tf.Tensor3D|ImageData|HTMLVideoElement|
//                                   HTMLImageElement|
//                                   HTMLCanvasElement): [number, number] {
//   return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
//                                       [input.height, input.width];
// }

export class HandDetector {
  private model: tfconv.GraphModel;
  private width: number;
  private height: number;
  private iouThreshold: number;
  private scoreThreshold: number;

  private anchors: Array<[number, number]>;
  private anchorsTensor: tf.Tensor2D;
  private inputSizeTensor: tf.Tensor1D;
  private doubleInputSizeTensor: tf.Tensor1D;

  constructor(
      model: tfconv.GraphModel, width: number, height: number,
      ANCHORS: Array<{x_center: number, y_center: number}>,
      iouThreshold: number, scoreThreshold: number) {
    this.model = model;
    this.width = width;
    this.height = height;
    this.iouThreshold = iouThreshold;
    this.scoreThreshold = scoreThreshold;

    this.anchors = ANCHORS.map(anchor => {
      const coord: [number, number] = [anchor.x_center, anchor.y_center];
      return coord;
    });
    this.anchorsTensor = tf.tensor2d(this.anchors);
    this.inputSizeTensor = tf.tensor1d([width, height]);
    this.doubleInputSizeTensor = tf.tensor1d([width * 2, height * 2]);
  }

  normalizeBoxes(boxes: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      const boxOffsets = tf.slice(boxes, [0, 0], [-1, 2]);
      const boxSizes = tf.slice(boxes, [0, 2], [-1, 2]);

      const boxCenterPoints =
          tf.add(tf.div(boxOffsets, this.inputSizeTensor), this.anchorsTensor);
      const halfBoxSizes = tf.div(boxSizes, this.doubleInputSizeTensor);

      const startPoints: tf.Tensor2D =
          tf.mul(tf.sub(boxCenterPoints, halfBoxSizes), this.inputSizeTensor);
      const endPoints: tf.Tensor2D =
          tf.mul(tf.add(boxCenterPoints, halfBoxSizes), this.inputSizeTensor);
      return tf.concat2d([startPoints, endPoints], 1);
    });
  }

  normalizeLandmarks(rawLandmarks: tf.Tensor2D, index: number): tf.Tensor2D {
    return tf.tidy(() => {
      const landmarks = tf.add(
          tf.div(rawLandmarks.reshape([-1, 7, 2]), this.inputSizeTensor),
          this.anchors[index]);

      return tf.mul(landmarks, this.inputSizeTensor);
    });
  }

  getBoundingBoxes(input: tf.Tensor4D) {
    return tf.tidy(() => {
      const normalizedInput = tf.mul(tf.sub(input, 0.5), 2);

      // The model returns a tensor with the following shape:
      //  [1 (batch), 2944 (anchor points), 19 (data for each anchor)]
      // Squeezing immediately because we are not batching inputs.
      const prediction: tf.Tensor2D =
          (this.model.predict(normalizedInput) as tf.Tensor3D).squeeze();

      // Regression score for each anchor point.
      const scores: tf.Tensor1D =
          tf.sigmoid(tf.slice(prediction, [0, 0], [-1, 1])).squeeze();

      // Bounding box for each anchor point.
      const rawBoxes = tf.slice(prediction, [0, 1], [-1, 4]);
      const boxes = this.normalizeBoxes(rawBoxes);

      const savedConsoleWarnFn = console.warn;
      console.warn = () => {};
      const boxesWithHands =
          tf.image
              .nonMaxSuppression(
                  boxes, scores, 1, this.iouThreshold, this.scoreThreshold)
              .arraySync();
      console.warn = savedConsoleWarnFn;

      if (boxesWithHands.length === 0) {
        return [null, null];
      }

      const boxIndex = boxesWithHands[0];
      const matchingBox = tf.slice(boxes, [boxIndex, 0], [1, -1]);

      const rawLandmarks = tf.slice(prediction, [boxIndex, 5], [1, 14]);
      const landmarks =
          this.normalizeLandmarks(rawLandmarks, boxIndex).reshape([-1, 2]);

      return [matchingBox, landmarks];
    });
  }

  estimateHandBounds(input: tf.Tensor4D) {
    const original_h = input.shape[1];
    const original_w = input.shape[2];

    const image: tf.Tensor4D =
        tf.tidy(() => input.resizeBilinear([this.width, this.height]).div(255));
    const bboxes_data = this.getBoundingBoxes(image);

    if (!bboxes_data[0]) {
      return null;
    }

    const bboxes = bboxes_data[0].arraySync() as any;
    const landmarks = bboxes_data[1].arraySync() as any;

    const factors: [number, number] =
        [original_w / this.width, original_h / this.height];

    const bb = scaleBoxCoordinates(
        {
          startPoint: bboxes[0].slice(0, 2),
          endPoint: bboxes[0].slice(2, 4),
          landmarks
        },
        factors);

    image.dispose();
    bboxes_data[0].dispose();
    bboxes_data[1].dispose();

    return bb;
  }
}
