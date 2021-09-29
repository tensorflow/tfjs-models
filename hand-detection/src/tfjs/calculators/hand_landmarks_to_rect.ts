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

  // Find boundaries of landmarks.
  let minX = Math.min(...landmarks.map(kpt => kpt.x));
  let maxX = Math.max(...landmarks.map(kpt => kpt.x));
  let minY = Math.min(...landmarks.map(kpt => kpt.y));
  let maxY = Math.max(...landmarks.map(kpt => kpt.y));

  const axisAlignedcenterX = (maxX + minX) / 2;
  const axisAlignedcenterY = (maxY + minY) / 2;

  // Find boundaries of rotated landmarks.
  const projectedXY = landmarks.map(kpt => {
    const originalX = (kpt.x - axisAlignedcenterX) * imageSize.width;
    const originalY = (kpt.y - axisAlignedcenterY) * imageSize.height;

    const projectedX =
        originalX * Math.cos(reverseAngle) - originalY * Math.sin(reverseAngle);
    const projectedY =
        originalX * Math.sin(reverseAngle) + originalY * Math.cos(reverseAngle);

    return [projectedX, projectedY];
  });

  minX = Math.min(...projectedXY.map(xy => xy[0]));
  maxX = Math.max(...projectedXY.map(xy => xy[0]));
  minY = Math.min(...projectedXY.map(xy => xy[1]));
  maxY = Math.max(...projectedXY.map(xy => xy[1]));

  const projectedcenterX = (maxX + minX) / 2;
  const projectedcenterY = (maxY + minY) / 2;

  const centerX = projectedcenterX * Math.cos(rotation) -
      projectedcenterY * Math.sin(rotation) +
      imageSize.width * axisAlignedcenterX;
  const centerY = projectedcenterX * Math.sin(rotation) +
      projectedcenterY * Math.cos(rotation) +
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
