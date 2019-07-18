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

import {convexHull} from './convexHull';
import {Point, Vector} from './geometry';
import {Box} from './types';

function dot(first: number[], second: number[]) {
  const add = (firstAddend: number, secondAddend: number) => {
    return firstAddend + secondAddend;
  };
  const mul = (_: number, idx: number) => {
    return first[idx] * second[idx];
  };
  if (first.length !== second.length) {
    throw Error('The arrays must have the same length');
  }
  return first.map(mul).reduce(add, 0);
}

type RotationMatrix = [[number, number], [number, number]];
const computeBackwardsRotationMatrix = (angle: number): RotationMatrix => [
    [Math.cos(angle), Math.cos(angle - Math.PI / 2)],
    [Math.cos(angle + Math.PI / 2), Math.cos(angle)]];

const inverseRotation = (rotation: RotationMatrix): RotationMatrix => {
  return [[rotation[0][0], rotation[1][0]], [rotation[0][1], rotation[1][1]]];
};

const rotate = (rotation: RotationMatrix, vector: Vector): Vector => {
  return rotation.map(row => {
    return dot(row, vector);
  }) as Vector;
};

export function minAreaRect(points: Point[]): Box {
  const convexHullPoints = convexHull(points);
  const edgeAngles = new Set<number>();
  for (let idx = 0; idx < convexHullPoints.length - 1; ++idx) {
    const edgeX = convexHullPoints[idx + 1].x - convexHullPoints[idx].x;
    const edgeY = convexHullPoints[idx + 1].y - convexHullPoints[idx].y;
    edgeAngles.add(Math.abs(Math.atan2(edgeY, edgeX) % (Math.PI / 2)));
  }
  // rotation angle, area, min x, max x, min y, max y
  let minBoundingBox = Array.from(new Array<number>(6), () => 0);
  // area
  minBoundingBox[1] = Number.MAX_VALUE;
  for (const angle of Array.from(edgeAngles)) {
    const backwardsRotation = computeBackwardsRotationMatrix(angle);
    const rotatedConvexHull = convexHullPoints.map(point => {
      const rotatedVector = rotate(backwardsRotation, [point.x, point.y]);
      return new Point(rotatedVector[0], rotatedVector[1]);
    });
    let minX = Number.MAX_VALUE;
    let maxX = Number.MIN_VALUE;
    let minY = Number.MAX_VALUE;
    let maxY = Number.MIN_VALUE;
    for (const point of rotatedConvexHull) {
      const {x, y} = point;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const width = maxX - minX;
    const height = maxY - minY;
    const area = width * height;
    if (area < minBoundingBox[1]) {
      minBoundingBox = [angle, area, minX, maxX, minY, maxY];
    }
  }
  const [angle, , minX, maxX, minY, maxY] = minBoundingBox;
  const backwardsRotation = computeBackwardsRotationMatrix(angle);
  const box: Box = [
    new Point(...rotate(inverseRotation(backwardsRotation), [maxX, maxY])),
    new Point(...rotate(inverseRotation(backwardsRotation), [minX, maxY])),
    new Point(...rotate(inverseRotation(backwardsRotation), [minX, minY])),
    new Point(...rotate(inverseRotation(backwardsRotation), [maxX, minY])),
  ];
  return box;
}
