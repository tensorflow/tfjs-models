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

// Container for the coordinates of the facial bounding box.
export type Box = {
  startEndTensor: tf.Tensor2D,
  startPoint: tf.Tensor2D,
  endPoint: tf.Tensor2D
};

export const disposeBox = (box: Box): void => {
  box.startEndTensor.dispose();
  box.startPoint.dispose();
  box.endPoint.dispose();
};

export const createBox = (startEndTensor: tf.Tensor2D): Box => ({
  startEndTensor,
  startPoint: tf.slice(startEndTensor, [0, 0], [-1, 2]),
  endPoint: tf.slice(startEndTensor, [0, 2], [-1, 2])
});

export const scaleBox = (box: Box, factors: tf.Tensor1D|[number, number]) => {
  const starts = tf.mul(box.startPoint, factors);
  const ends = tf.mul(box.endPoint, factors);

  const newCoordinates =
      tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);

  return createBox(newCoordinates);
};
