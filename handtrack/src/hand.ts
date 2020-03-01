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

export class HandDetector {
  private model: tfconv.GraphModel;
  private anchors: tf.Tensor;
  private input_size: tf.Tensor;
  private width: number;
  private height: number;
  private iouThreshold: number;
  private scoreThreshold: number;

  constructor(
      model: tfconv.GraphModel, width: number, height: number, ANCHORS: any,
      iouThreshold: number, scoreThreshold: number) {
    this.model = model;
    this.anchors = this._generate_anchors(ANCHORS);
    this.width = width;
    this.height = height;
    this.input_size = tf.tensor([width, height]);

    this.iouThreshold = iouThreshold;
    this.scoreThreshold = scoreThreshold;
  }

  _generate_anchors(ANCHORS: any) {
    const anchors = [];

    for (let i = 0; i < ANCHORS.length; ++i) {
      const anchor = ANCHORS[i];
      anchors.push([anchor.x_center, anchor.y_center]);
    }
    return tf.tensor(anchors);
  }

  _decode_bounds(box_outputs: tf.Tensor) {
    const box_starts = tf.slice(box_outputs, [0, 0], [-1, 2]);
    const centers = tf.add(tf.div(box_starts, this.input_size), this.anchors);
    const box_sizes = tf.slice(box_outputs, [0, 2], [-1, 2]);

    const box_sizes_norm = tf.div(box_sizes, this.input_size);
    const halfBoxSize = tf.div(box_sizes_norm, 2);

    const starts = tf.sub(centers, halfBoxSize);
    const ends = tf.add(centers, halfBoxSize);

    return tf.concat2d(
        [
          tf.mul(starts as tf.Tensor2D, this.input_size as tf.Tensor2D) as
              tf.Tensor2D,
          tf.mul(ends, this.input_size) as tf.Tensor2D
        ],
        1);
  }

  _decode_landmarks(raw_landmarks: tf.Tensor) {
    const relative_landmarks = tf.add(
        tf.div(raw_landmarks.reshape([-1, 7, 2]), this.input_size),
        this.anchors.reshape([-1, 1, 2]));

    return tf.mul(relative_landmarks, this.input_size);
  }

  _getBoundingBox(input_image: tf.Tensor) {
    return tf.tidy(() => {
      const img = tf.mul(tf.sub(input_image, 0.5), 2);  // make input [-1, 1]

      const prediction = this.model.predict(img) as tf.Tensor;

      const scores =
          tf.sigmoid(tf.slice(prediction, [0, 0, 0], [1, -1, 1])).squeeze() as
          tf.Tensor1D;

      const raw_boxes = tf.slice(prediction, [0, 0, 1], [1, -1, 4]).squeeze();
      const raw_landmarks =
          tf.slice(prediction, [0, 0, 5], [1, -1, 14]).squeeze();
      const boxes = this._decode_bounds(raw_boxes);

      const box_indices =
          tf.image
              .nonMaxSuppression(
                  boxes, scores, 1, this.iouThreshold, this.scoreThreshold)
              .arraySync();

      const landmarks = this._decode_landmarks(raw_landmarks);
      if (box_indices.length == 0) {
        return [null, null];
      }

      const box_index = box_indices[0];
      const result_box = tf.slice(boxes, [box_index, 0], [1, -1]);

      const result_landmarks =
          tf.slice(landmarks, [box_index, 0], [1]).reshape([-1, 2]);

      return [result_box, result_landmarks];
    });
  }

  getSingleBoundingBox(input_image: tf.Tensor4D) {
    const original_h = input_image.shape[1];
    const original_w = input_image.shape[2];

    const image =
        tf.tidy(() => input_image.resizeBilinear([256, 256]).div(255));
    const bboxes_data = this._getBoundingBox(image);

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
