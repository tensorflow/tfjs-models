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

import {Point} from './geometry';

export function isGreater(first: Point, second: Point): number {
  return [first.x, first.y].join('').localeCompare(
      [second.x, second.y].join(''));
}

export function monotoneChain(points: Point[]): Point[] {
  // assumes that points are lexicographically ordered
  if (points.length <= 1) {
    return points;
  }

  const upperHull: Point[] = [];
  points.forEach((point) => {
    while (upperHull.length >= 2) {
      const lastCandidate = upperHull[upperHull.length - 1];
      const oneBeforeLastCandidate = upperHull[upperHull.length - 2];
      //  the sequence of last two points in upperHull and the point do not make
      //  a counter-clockwise turn
      if ((lastCandidate.x - oneBeforeLastCandidate.x) *
              (point.y - oneBeforeLastCandidate.y) >=
          (lastCandidate.y - oneBeforeLastCandidate.y) *
              (point.x - oneBeforeLastCandidate.x)) {
        upperHull.pop();
      } else {
        break;
      }
    }
    upperHull.push(point);
  });
  upperHull.pop();

  const lowerHull: Point[] = [];
  for (let idx = points.length - 1; idx >= 0; --idx) {
    const point = points[idx];
    while (lowerHull.length >= 2) {
      const lastCandidate = lowerHull[lowerHull.length - 1];
      const oneBeforeLastCandidate = lowerHull[lowerHull.length - 2];
      if ((lastCandidate.x - oneBeforeLastCandidate.x) *
              (point.y - oneBeforeLastCandidate.y) >=
          (lastCandidate.y - oneBeforeLastCandidate.y) *
              (point.x - oneBeforeLastCandidate.x)) {
        lowerHull.pop();
      } else {
        break;
      }
    }
    lowerHull.push(point);
  }
  lowerHull.pop();

  if (upperHull.length === 1 && lowerHull.length === 1 &&
      upperHull[0].x === lowerHull[0].x && upperHull[0].y === lowerHull[0].y) {
    return upperHull;
  } else {
    return upperHull.concat(lowerHull);
  }
}

export function convexHull(points: Point[]): Point[] {
  points.sort(isGreater);
  return monotoneChain(points);
}
