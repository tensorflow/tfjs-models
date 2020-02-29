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

import * as tf from '@tensorflow/tfjs-core';

// The facial bounding box.
export type Box = {
  startPoint: tf.Tensor2D,     // Upper left hand corner of bounding box.
  endPoint: tf.Tensor2D,       // Lower right hand corner of bounding box.
  startEndTensor: tf.Tensor2D  // Concatenation of start and end points.
};

export function disposeBox(box: Box): void {
  if (box != null && box.startPoint != null) {
    box.startEndTensor.dispose();
    box.startPoint.dispose();
    box.endPoint.dispose();
  }
}

export function createBox(
    startEndTensor: tf.Tensor2D, startPoint?: tf.Tensor2D,
    endPoint?: tf.Tensor2D): Box {
  return {
    startEndTensor,
    startPoint: startPoint != null ? startPoint :
                                     tf.slice(startEndTensor, [0, 0], [-1, 2]),
    endPoint: endPoint != null ? endPoint :
                                 tf.slice(startEndTensor, [0, 2], [-1, 2])
  };
}

export function scaleBoxCoordinates(
    box: Box, factor: tf.Tensor1D|[number, number]): Box {
  const newStart: tf.Tensor2D = tf.mul(box.startPoint, factor);
  const newEnd: tf.Tensor2D = tf.mul(box.endPoint, factor);

  return createBox(tf.concat2d([newStart, newEnd], 1));
}

export function getBoxSize(box: Box): tf.Tensor2D {
  return tf.tidy(() => {
    const diff: tf.Tensor2D = tf.sub(box.endPoint, box.startPoint);
    return tf.abs(diff);
  });
}

export function getBoxCenter(box: Box): tf.Tensor2D {
  return tf.tidy(() => {
    const halfSize = tf.div(tf.sub(box.endPoint, box.startPoint), 2);
    return tf.add(box.startPoint, halfSize);
  });
}

export function cutBoxFromImageAndResize(
    box: Box, image: tf.Tensor4D, cropSize: [number, number]): tf.Tensor4D {
  const height = image.shape[1];
  const width = image.shape[2];
  const xyxy = box.startEndTensor;

  return tf.tidy(() => {
    const yxyx = tf.concat2d(
        [
          xyxy.slice([0, 1], [-1, 1]), xyxy.slice([0, 0], [-1, 1]),
          xyxy.slice([0, 3], [-1, 1]), xyxy.slice([0, 2], [-1, 1])
        ],
        0);
    const roundedCoords: tf.Tensor2D =
        tf.div(yxyx.transpose(), [height, width, height, width]);
    return tf.image.cropAndResize(image, roundedCoords, [0], cropSize);
  });
}

export function enlargeBox(box: Box, factor = 1.5): Box {
  return tf.tidy(() => {
    const center = getBoxCenter(box);
    const size = getBoxSize(box);
    const newSize = tf.mul(tf.div(size, 2), factor);
    const newStart: tf.Tensor2D = tf.sub(center, newSize);
    const newEnd: tf.Tensor2D = tf.add(center, newSize);

    return createBox(tf.concat2d([newStart, newEnd], 1), newStart, newEnd);
  });
}
