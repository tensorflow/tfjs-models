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
  startEndTensor: tf.Tensor2D,
  startPoint: tf.Tensor2D,  // Upper left hand corner of bounding box.
  endPoint: tf.Tensor2D     // Lower right hand corner of bounding box.
};

const getBoxCenter = (box: Box): tf.Tensor2D => {
  const halfSize = tf.div(tf.sub(box.endPoint, box.startPoint), 2);
  return tf.add(box.startPoint, halfSize);
};

export function disposeBox(box: Box): void {
  box.startEndTensor.dispose();
  box.startPoint.dispose();
  box.endPoint.dispose();
}

export function createBox(
    startEndTensor: tf.Tensor2D, startPoint?: tf.Tensor2D,
    endPoint?: tf.Tensor2D): Box {
  return {
    startEndTensor,
    startPoint: startPoint ? startPoint :
                             tf.slice(startEndTensor, [0, 0], [-1, 2]),
    endPoint: endPoint ? endPoint : tf.slice(startEndTensor, [0, 2], [-1, 2])
  };
}

export function scaleBoxCoordinates(
    box: Box, factor: tf.Tensor1D|[number, number]): Box {
  const start: tf.Tensor2D = tf.mul(box.startPoint, factor);
  const end: tf.Tensor2D = tf.mul(box.endPoint, factor);

  return createBox(tf.concat2d([start, end], 1), start, end);
}

export function enlargeBox(box: Box, factor = 1.5) {
  const center = getBoxCenter(box);
  const size = getBoxSize(box);
  const newSize = tf.mul(tf.div(size, 2), factor);
  const newStart: tf.Tensor2D = tf.sub(center, newSize);
  const newEnd: tf.Tensor2D = tf.add(center, newSize);

  return createBox(tf.concat2d([newStart, newEnd], 1), newStart, newEnd);
}

export function getBoxSize(box: Box): tf.Tensor2D {
  return tf.abs(tf.sub(box.endPoint, box.startPoint)) as tf.Tensor2D;
}

export function cutBoxFromImageAndResize(
    box: Box, image: tf.Tensor4D, cropSize: [number, number]): tf.Tensor4D {
  const h = image.shape[1];
  const w = image.shape[2];

  const xyxy = box.startEndTensor.arraySync()[0];
  const yxyx = [xyxy[1], xyxy[0], xyxy[3], xyxy[2]];
  const roundedCoords = [yxyx[0] / h, yxyx[1] / w, yxyx[2] / h, yxyx[3] / w];
  return tf.image.cropAndResize(image, [roundedCoords], [0], cropSize);
}
