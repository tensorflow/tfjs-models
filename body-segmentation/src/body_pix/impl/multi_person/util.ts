/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {NUM_KEYPOINTS} from '../keypoints';
import {Padding, Part, TensorBuffer3D, Vector2D} from '../types';

export function getScale(
    [height, width]: [number, number],
    [inputResolutionY, inputResolutionX]: [number, number],
    padding: Padding): [number, number] {
  const {top: padT, bottom: padB, left: padL, right: padR} = padding;
  const scaleY = inputResolutionY / (padT + padB + height);
  const scaleX = inputResolutionX / (padL + padR + width);
  return [scaleX, scaleY];
}

export function getOffsetPoint(
    y: number, x: number, keypoint: number, offsets: TensorBuffer3D): Vector2D {
  return {
    y: offsets.get(y, x, keypoint),
    x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}

export function getImageCoords(
    part: Part, outputStride: number, offsets: TensorBuffer3D): Vector2D {
  const {heatmapY, heatmapX, id: keypoint} = part;
  const {y, x} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets);
  return {
    x: part.heatmapX * outputStride + x,
    y: part.heatmapY * outputStride + y
  };
}

export function fillArray<T>(element: T, size: number): T[] {
  const result: T[] = new Array(size);

  for (let i = 0; i < size; i++) {
    result[i] = element;
  }

  return result;
}

export function clamp(a: number, min: number, max: number): number {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

export function squaredDistance(
    y1: number, x1: number, y2: number, x2: number): number {
  const dy = y2 - y1;
  const dx = x2 - x1;
  return dy * dy + dx * dx;
}

export function addVectors(a: Vector2D, b: Vector2D): Vector2D {
  return {x: a.x + b.x, y: a.y + b.y};
}

export function clampVector(a: Vector2D, min: number, max: number): Vector2D {
  return {y: clamp(a.y, min, max), x: clamp(a.x, min, max)};
}
