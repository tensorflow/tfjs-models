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
  startPoint: tf.Tensor2D,      // Upper left hand corner of bounding box.
  endPoint: tf.Tensor2D,        // Lower right hand corner of bounding box.
  startEndTensor: tf.Tensor2D,  // Concatenation of start and end points.
  landmarks?: any
};

export type cpuBox = {
  startPoint: [number, number],  // Upper left hand corner of bounding box.
  endPoint: [number, number],    // Lower right hand corner of bounding box.
  landmarks?: Array<[number, number]>
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
    box: cpuBox, factor: [number, number]): cpuBox {
  const startPoint: [number, number] =
      [box.startPoint[0] * factor[0], box.startPoint[1] * factor[1]];
  const endPoint: [number, number] =
      [box.endPoint[0] * factor[0], box.endPoint[1] * factor[1]];

  return {startPoint, endPoint};
}

export function getBoxSize(box: cpuBox): [number, number] {
  return [
    Math.abs(box.endPoint[0] - box.startPoint[0]),
    Math.abs(box.endPoint[1] - box.startPoint[1])
  ];
}

export function getBoxCenter(box: cpuBox): [number, number] {
  return [
    box.startPoint[0] + (box.endPoint[0] - box.startPoint[0]) / 2,
    box.startPoint[1] + (box.endPoint[1] - box.startPoint[1]) / 2
  ];
}

export function cutBoxFromImageAndResize(
    box: cpuBox, image: tf.Tensor4D, cropSize: [number, number]): tf.Tensor4D {
  const h = image.shape[1];
  const w = image.shape[2];

  const boxes = [[
    box.startPoint[1] / h, box.startPoint[0] / w, box.endPoint[1] / h,
    box.endPoint[0] / w
  ]];

  return tf.image.cropAndResize(image, boxes, [0], cropSize);
}

export function enlargeBox(box: Box, factor = 1.5): Box {
  return tf.tidy(() => {
    const boxCPU = {
      startPoint: box.startPoint.dataSync() as {} as [number, number],
      endPoint: box.endPoint.dataSync() as {} as [number, number]
    };
    const center = getBoxCenter(boxCPU);
    const size = tf.tensor2d(getBoxSize(boxCPU), [1, 2]);
    const newSize = tf.mul(tf.div(size, 2), factor);
    const newStart: tf.Tensor2D = tf.sub(center, newSize);
    const newEnd: tf.Tensor2D = tf.add(center, newSize);

    return createBox(tf.concat2d([newStart, newEnd], 1), newStart, newEnd);
  });
}
