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

import {Coord2D, Coords3D} from './util';

// The facial bounding box.
export type Box = {
  startPoint: Coord2D,  // Upper left hand corner of bounding box.
  endPoint: Coord2D,    // Lower right hand corner of bounding box.
  landmarks?: Coords3D
};

export function scaleBoxCoordinates(box: Box, factor: Coord2D): Box {
  const startPoint: Coord2D =
      [box.startPoint[0] * factor[0], box.startPoint[1] * factor[1]];
  const endPoint: Coord2D =
      [box.endPoint[0] * factor[0], box.endPoint[1] * factor[1]];

  return {startPoint, endPoint};
}

export function getBoxSize(box: Box): Coord2D {
  return [
    Math.abs(box.endPoint[0] - box.startPoint[0]),
    Math.abs(box.endPoint[1] - box.startPoint[1])
  ];
}

export function getBoxCenter(box: Box): Coord2D {
  return [
    box.startPoint[0] + (box.endPoint[0] - box.startPoint[0]) / 2,
    box.startPoint[1] + (box.endPoint[1] - box.startPoint[1]) / 2
  ];
}

export function cutBoxFromImageAndResize(
    box: Box, image: tf.Tensor4D, cropSize: Coord2D): tf.Tensor4D {
  const h = image.shape[1];
  const w = image.shape[2];

  const boxes = [[
    box.startPoint[1] / h, box.startPoint[0] / w, box.endPoint[1] / h,
    box.endPoint[0] / w
  ]];

  return tf.image.cropAndResize(image, boxes, [0], cropSize,
    'bilinear' /* method */,
    0 /* extrapolation value */);
}

/**
 * Enlarges the box by the provided factor.
 * @param box An object with startPoint and endPoint properties describing the
 * outlines of the box to be enlarged.
 * @param factor optional The enlargement factor. Defaults to 1.5
 */
export function enlargeBox(box: Box, factor = 1.5): Box {
  const center = getBoxCenter(box);
  const size = getBoxSize(box);

  const newHalfSize = [factor * size[0] / 2, factor * size[1] / 2];
  const startPoint: Coord2D =
      [center[0] - newHalfSize[0], center[1] - newHalfSize[1]];
  const endPoint: Coord2D =
      [center[0] + newHalfSize[0], center[1] + newHalfSize[1]];

  return {startPoint, endPoint, landmarks: box.landmarks};
}

/**
 * Squarifies the provided box by setting its length and height equal to
 * max(length, height) while preserving its center point.
 * @param box An object with startPoint and endPoint properties describing the
 * outlines of the box to be squarified.
 */
export function squarifyBox(box: Box): Box {
  const centers = getBoxCenter(box);
  const size = getBoxSize(box);
  const maxEdge = Math.max(...size);

  const halfSize = maxEdge / 2;
  const startPoint: [number, number] =
      [centers[0] - halfSize, centers[1] - halfSize];
  const endPoint: [number, number] =
      [centers[0] + halfSize, centers[1] + halfSize];

  return {startPoint, endPoint, landmarks: box.landmarks};
}
