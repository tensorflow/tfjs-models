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

// MIT License
// Copyright (c) 2017 Vincent Mühler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {Moments} from './Moments';
import {Point2} from './Point2';
import {Rect} from './Rect';
import {RotatedRect} from './RotatedRect';
import {Vec4} from './Vec4';

export class Contour {
  readonly numPoints: number;
  readonly area: number;
  readonly isConvex: boolean;
  readonly hierarchy: Vec4;
  constructor();
  constructor(pts: Point2[]);
  constructor(pts: number[][]);
  approxPolyDP(epsilon: number, closed: boolean): Point2[];
  approxPolyDPContour(epsilon: number, closed: boolean): Contour;
  arcLength(closed?: boolean): number;
  boundingRect(): Rect;
  convexHull(clockwise?: boolean): Contour;
  convexHullIndices(clockwise?: boolean): number[];
  convexityDefects(hullIndices: number[]): Vec4[];
  fitEllipse(): RotatedRect;
  getPoints(): Point2[];
  matchShapes(contour2: Contour, method: number): number;
  minAreaRect(): RotatedRect;
  minEnclosingCircle(): {center: Point2, radius: number};
  minEnclosingTriangle(): Point2[];
  moments(): Moments;
  pointPolygonTest(pt: Point2): number;
}
