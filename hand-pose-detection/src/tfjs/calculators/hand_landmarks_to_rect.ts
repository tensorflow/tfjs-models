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

import {normalizeRadians} from '../../shared/calculators/image_utils';
import {ImageSize, Keypoint} from '../../shared/calculators/interfaces/common_interfaces';
import {Rect} from '../../shared/calculators/interfaces/shape_interfaces';

const WRIST_JOINT = 0;
const MIDDLE_FINGER_PIP_JOINT = 6;
const INDEX_FINGER_PIP_JOINT = 4;
const RING_FINGER_PIP_JOINT = 8;

function computeRotation(landmarks: Keypoint[], imageSize: ImageSize) {
  const x0 = landmarks[WRIST_JOINT].x * imageSize.width;
  const y0 = landmarks[WRIST_JOINT].y * imageSize.height;

  let x1 = (landmarks[INDEX_FINGER_PIP_JOINT].x +
            landmarks[RING_FINGER_PIP_JOINT].x) /
      2;
  let y1 = (landmarks[INDEX_FINGER_PIP_JOINT].y +
            landmarks[RING_FINGER_PIP_JOINT].y) /
      2;
  x1 = (x1 + landmarks[MIDDLE_FINGER_PIP_JOINT].x) / 2 * imageSize.width;
  y1 = (y1 + landmarks[MIDDLE_FINGER_PIP_JOINT].y) / 2 * imageSize.height;

  const rotation =
      normalizeRadians(Math.PI / 2 - Math.atan2(-(y1 - y0), x1 - x0));
  return rotation;
}

/**
 * @param landmarks List of normalized landmarks.
 *
 * @returns A `Rect`.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc;bpv=1
export function handLandmarksToRect(
    landmarks: Keypoint[], imageSize: ImageSize): Rect {
  const rotation = computeRotation(landmarks, imageSize);
  const reverseAngle = normalizeRadians(-rotation);

  let minX = Number.POSITIVE_INFINITY, maxX = Number.NEGATIVE_INFINITY,
      minY = Number.POSITIVE_INFINITY, maxY = Number.NEGATIVE_INFINITY;

  // Find boundaries of landmarks.
  for (const landmark of landmarks) {
    const x = landmark.x;
    const y = landmark.y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const axisAlignedcenterX = (maxX + minX) / 2;
  const axisAlignedcenterY = (maxY + minY) / 2;

  minX = Number.POSITIVE_INFINITY, maxX = Number.NEGATIVE_INFINITY,
  minY = Number.POSITIVE_INFINITY, maxY = Number.NEGATIVE_INFINITY;
  // Find boundaries of rotated landmarks.
  for (const landmark of landmarks) {
    const originalX = (landmark.x - axisAlignedcenterX) * imageSize.width;
    const originalY = (landmark.y - axisAlignedcenterY) * imageSize.height;

    const projectedX =
        originalX * Math.cos(reverseAngle) - originalY * Math.sin(reverseAngle);
    const projectedY =
        originalX * Math.sin(reverseAngle) + originalY * Math.cos(reverseAngle);

    minX = Math.min(minX, projectedX);
    maxX = Math.max(maxX, projectedX);
    minY = Math.min(minY, projectedY);
    maxY = Math.max(maxY, projectedY);
  }

  const projectedCenterX = (maxX + minX) / 2;
  const projectedCenterY = (maxY + minY) / 2;

  const centerX = projectedCenterX * Math.cos(rotation) -
      projectedCenterY * Math.sin(rotation) +
      imageSize.width * axisAlignedcenterX;
  const centerY = projectedCenterX * Math.sin(rotation) +
      projectedCenterY * Math.cos(rotation) +
      imageSize.height * axisAlignedcenterY;
  const width = (maxX - minX) / imageSize.width;
  const height = (maxY - minY) / imageSize.height;

  return {
    xCenter: centerX / imageSize.width,
    yCenter: centerY / imageSize.height,
    width,
    height,
    rotation
  };
}
