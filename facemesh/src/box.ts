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

import * as tf from '@tensorflow/tfjs-core';

// The bounding box of the face mesh.
export class Box {
  public startEndTensor: tf.Tensor2D;
  public startPoint: tf.Tensor2D;
  public endPoint: tf.Tensor2D;

  constructor(startEndTensor: tf.Tensor2D) {
    this.startEndTensor = startEndTensor;
    this.startPoint = tf.slice(startEndTensor, [0, 0], [-1, 2]);
    this.endPoint = tf.slice(startEndTensor, [0, 2], [-1, 2]);
  }

  getSize(): tf.Tensor2D {
    return tf.abs(tf.sub(this.endPoint, this.startPoint)) as tf.Tensor2D;
  }

  getCenter(): tf.Tensor2D {
    const halfSize = tf.div(tf.sub(this.endPoint, this.startPoint), 2);
    return tf.add(this.startPoint, halfSize);
  }

  cutFromAndResize(image: tf.Tensor4D, cropSize: [number, number]):
      tf.Tensor4D {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    const yxyx = tf.concat2d(
        [
          xyxy.slice([0, 1], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 0], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 3], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 2], [-1, 1]) as tf.Tensor2D
        ],
        0);
    const roundedCoords = tf.div(yxyx.transpose(), [h, w, h, w]);
    return tf.image.cropAndResize(
        image, roundedCoords as tf.Tensor2D, [0], cropSize);
  }

  scale(factors: tf.Tensor1D): Box {
    const starts = tf.mul(this.startPoint, factors);
    const ends = tf.mul(this.endPoint, factors);

    const newCoordinates =
        tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);
    return new Box(newCoordinates);
  }

  increaseBox(ratio = 1.5) {
    const center = this.getCenter();
    const size = this.getSize();
    const newSize = tf.mul(tf.div(size, 2), ratio);
    const newStart = tf.sub(center, newSize);
    const newEnd = tf.add(center, newSize);

    this.startEndTensor =
        tf.concat2d([newStart as tf.Tensor2D, newEnd as tf.Tensor2D], 1);
    this.startPoint = newStart as tf.Tensor2D;
    this.endPoint = newEnd as tf.Tensor2D;
  }
}
