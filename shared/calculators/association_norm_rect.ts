/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {BoundingBox, Rect} from './interfaces/shape_interfaces';

function area(rect: BoundingBox) {
  return rect.width * rect.height;
}

function intersects(rect1: BoundingBox, rect2: BoundingBox) {
  return !(
      rect1.xMax < rect2.xMin || rect2.xMax < rect1.xMin ||
      rect1.yMax < rect2.yMin || rect2.yMax < rect1.yMin);
}

function intersect(rect1: BoundingBox, rect2: BoundingBox) {
  const xMin = Math.max(rect1.xMin, rect2.xMin);
  const xMax = Math.min(rect1.xMax, rect2.xMax);
  const yMin = Math.max(rect1.yMin, rect2.yMin);
  const yMax = Math.min(rect1.yMax, rect2.yMax);
  const width = Math.max(xMax - xMin, 0);
  const height = Math.max(yMax - yMin, 0);

  return {xMin, xMax, yMin, yMax, width, height};
}

export function getBoundingBox(rect: Rect): BoundingBox {
  const xMin = rect.xCenter - rect.width / 2;
  const xMax = xMin + rect.width;
  const yMin = rect.yCenter - rect.height / 2;
  const yMax = yMin + rect.height;
  return {xMin, xMax, yMin, yMax, width: rect.width, height: rect.height};
}

function overlapSimilarity(rect1: Rect, rect2: Rect): number {
  const bbox1 = getBoundingBox(rect1);
  const bbox2 = getBoundingBox(rect2);
  if (!intersects(bbox1, bbox2)) {
    return 0;
  }
  const intersectionArea = area(intersect(bbox1, bbox2));
  const normalization = area(bbox1) + area(bbox2) - intersectionArea;
  return normalization > 0 ? intersectionArea / normalization : 0;
}

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/association_norm_rect_calculator.cc
// Propgating ids from previous to current is not performed by this code.
export function calculateAssociationNormRect(
    rectsArray: Rect[][], minSimilarityThreshold: number): Rect[] {
  let result: Rect[] = [];

  // rectsArray elements are interpreted to be sorted in reverse priority order,
  // so later elements are higher in priority. This means that if there's a
  // large overlap, the later rect will be added and the older rect will be
  // removed.
  rectsArray.forEach(rects => rects.forEach(curRect => {
    result = result.filter(
        prevRect =>
            overlapSimilarity(curRect, prevRect) <= minSimilarityThreshold);
    result.push(curRect);
  }));

  return result;
}
