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

// startEndTensor: [1, 4]|[1, 6]
// startPoint: [1, 2]|[1, 3]
export const createBox =
    (startEndTensor: tf.Tensor2D, startPoint?: tf.Tensor2D,
     endPoint?: tf.Tensor2D): Box => {
      const dims = startEndTensor.shape[1] / 2;

      return {
        startEndTensor,
        startPoint: startPoint ? startPoint :
                                 tf.slice(startEndTensor, [0, 0], [-1, dims]),
        endPoint: endPoint ? endPoint :
                             tf.slice(startEndTensor, [0, dims], [-1, dims])
      };
    };

export const scaleBox = (box: Box, factors: tf.Tensor1D|number[]): Box => {
  if (factors instanceof tf.Tensor) {
    if (box.startPoint.shape[1] > factors.shape[0]) {
      factors = tf.concat([factors, tf.tensor1d([1])]);
    }
  } else {
    if (box.startPoint.shape[1] > factors.length) {
      factors = [...factors, 1];
    }
  }

  const starts = tf.mul(box.startPoint, factors);
  const ends = tf.mul(box.endPoint, factors);

  const newCoordinates =
      tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);

  return createBox(newCoordinates);
};

export const getBoxSize = (box: Box): tf.Tensor2D => {
  return tf.abs(tf.sub(box.endPoint, box.startPoint)) as tf.Tensor2D;
};

export const getBoxCenter = (box: Box): tf.Tensor2D => {
  const halfSize = tf.div(tf.sub(box.endPoint, box.startPoint), 2);
  return tf.add(box.startPoint, halfSize);
};

export const cutBoxFromImageAndResize =
    (box: Box, image: tf.Tensor4D, cropSize: [number, number]): tf.Tensor4D => {
      const h = image.shape[1];
      const w = image.shape[2];

      const xyxy = box.startEndTensor;  // TODO: 3D SUPPORT
      const dims = xyxy.shape[1] / 2;
      const yxyx = tf.concat2d(
          [
            xyxy.slice([0, 1], [-1, 1]) as tf.Tensor2D,
            xyxy.slice([0, 0], [-1, 1]) as tf.Tensor2D,
            xyxy.slice([0, dims + 1], [-1, 1]) as tf.Tensor2D,
            xyxy.slice([0, dims], [-1, 1]) as tf.Tensor2D
          ],
          0);
      const roundedCoords = tf.div(yxyx.transpose(), [h, w, h, w]);
      return tf.image.cropAndResize(
          image, roundedCoords as tf.Tensor2D, [0], cropSize);
    };

export const enlargeBox = (box: Box, factor = 1.5) => {
  const center = getBoxCenter(box);
  const size = getBoxSize(box);
  const newSize = tf.mul(tf.div(size, 2), factor);
  const newStart = tf.sub(center, newSize) as tf.Tensor2D;
  const newEnd = tf.add(center, newSize) as tf.Tensor2D;

  return createBox(tf.concat2d([newStart, newEnd], 1), newStart, newEnd);
};
