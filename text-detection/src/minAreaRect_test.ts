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
import {minAreaRect} from './minAreaRect';

describe('minAreaRect', () => {
  it('minAreaRect returns a line for collinear points.', () => {
    const points = [[1, 2], [3, 4], [4, 5], [6, 7], [8, 9]].map(
        coords => new Point(...(coords as [number, number])));
    const rect = minAreaRect(points);
    const [upright, upleft, downleft, downright] = rect;
    const left = (upleft.sub(downleft)).norm();
    const right = (upright.sub(downright)).norm();
    const up = (upright.sub(upleft)).norm();
    const down = (downright.sub(downleft)).norm();
    const epsilon = 1e-9;

    const isSensible =
        right - left < epsilon && up - down < epsilon && right * up < epsilon;
    expect(isSensible).toEqual(true);
  });
  it('minAreaRect returns the rectangle itself on the rectangle input', () => {
    const rawPoints = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
    const points =
        rawPoints.map(coords => new Point(...(coords as [number, number])));
    const rect = minAreaRect(points);
    const epsilon = 1e-9;
    expect(
        rect.map(
            (point, idx) => Math.abs(point.x - rawPoints[idx][0]) < epsilon &&
                Math.abs(point.y - rawPoints[idx][1]) < epsilon))
        .toEqual(Array.from(new Array(4), () => true));
  });
});
