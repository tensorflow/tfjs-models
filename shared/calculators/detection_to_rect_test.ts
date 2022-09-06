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

import {calculateDetectionsToRects} from './detection_to_rect';
import {Detection, Rect} from './interfaces/shape_interfaces';

function detectionWithLocationData(
    xMin: number, yMin: number, width: number, height: number) {
  const detection: Detection = {
    locationData: {
      boundingBox:
          {xMin, yMin, width, height, xMax: xMin + width, yMax: yMin + height}
    }
  };
  return detection;
}

function detectionWithKeyPoints(keypoints: Array<[number, number]>) {
  const detection: Detection = {
    locationData: {
      relativeKeypoints:
          keypoints.map(keypoint => ({x: keypoint[0], y: keypoint[1]}))
    }
  };
  return detection;
}

function detectionWithRelativeLocationData(
    xMin: number, yMin: number, width: number, height: number) {
  const detection: Detection = {
    locationData: {
      relativeBoundingBox:
          {xMin, yMin, width, height, xMax: xMin + width, yMax: yMin + height}
    }
  };
  return detection;
}

function expectRectEq(
    rect: Rect, xCenter: number, yCenter: number, width: number,
    height: number) {
  expect(rect.xCenter).toBe(xCenter);
  expect(rect.yCenter).toBe(yCenter);
  expect(rect.width).toBe(width);
  expect(rect.height).toBe(height);
}

describe('DetectionsToRects', () => {
  it('detection to rect.', async () => {
    const rects = calculateDetectionsToRects(
        detectionWithLocationData(100, 200, 300, 400), 'boundingbox', 'rect');

    expectRectEq(rects, 250, 400, 300, 400);
  });

  it('detection keypoints to rect.', async () => {
    let rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0, 0], [1, 1]]), 'keypoints', 'rect',
        {width: 640, height: 480});

    expectRectEq(rects, 320, 240, 640, 480);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0.25, 0.25], [0.75, 0.75]]), 'keypoints',
        'rect', {width: 640, height: 480});

    expectRectEq(rects, 320, 240, 320, 240);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0, 0], [0.5, 0.5]]), 'keypoints', 'rect',
        {width: 640, height: 480});

    expectRectEq(rects, 160, 120, 320, 240);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0.5, 0.5], [1, 1]]), 'keypoints', 'rect',
        {width: 640, height: 480});

    expectRectEq(rects, 480, 360, 320, 240);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0.25, 0.25], [0.75, 0.75]]), 'keypoints',
        'rect', {width: 0, height: 0});

    expectRectEq(rects, 0, 0, 0, 0);
  });

  it('detection to normalized rect.', async () => {
    const rects = calculateDetectionsToRects(
        detectionWithRelativeLocationData(0.1, 0.2, 0.3, 0.4), 'boundingbox',
        'normRect');

    expectRectEq(rects, 0.25, 0.4, 0.3, 0.4);
  });

  it('detection keypoints to normalized rect.', async () => {
    let rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0, 0], [0.5, 0.5], [1, 1]]), 'keypoints',
        'normRect');

    expectRectEq(rects, 0.5, 0.5, 1, 1);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]]),
        'keypoints', 'normRect');

    expectRectEq(rects, 0.5, 0.5, 0.5, 0.5);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0, 0], [0.5, 0.5]]), 'keypoints', 'normRect');

    expectRectEq(rects, 0.25, 0.25, 0.5, 0.5);

    rects = calculateDetectionsToRects(
        detectionWithKeyPoints([[0.5, 0.5], [1, 1]]), 'keypoints', 'normRect');

    expectRectEq(rects, 0.75, 0.75, 0.5, 0.5);
  });
});
