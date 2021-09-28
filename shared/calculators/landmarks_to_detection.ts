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

import {Keypoint} from './interfaces/common_interfaces';
import {Detection} from './interfaces/shape_interfaces';

/**
 * Converts normalized Landmark to `Detection`. A relative bounding box will
 * be created containing all landmarks exactly.
 * @param landmarks List of normalized landmarks.
 *
 * @returns A `Detection`.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_to_detection_calculator.cc
export function landmarksToDetection(landmarks: Keypoint[]): Detection {
  const detection:
      Detection = {locationData: {relativeKeypoints: [] as Keypoint[]}};

  let xMin = Number.MAX_SAFE_INTEGER;
  let xMax = Number.MIN_SAFE_INTEGER;
  let yMin = Number.MAX_SAFE_INTEGER;
  let yMax = Number.MIN_SAFE_INTEGER;

  for (let i = 0; i < landmarks.length; ++i) {
    const landmark = landmarks[i];
    xMin = Math.min(xMin, landmark.x);
    xMax = Math.max(xMax, landmark.x);
    yMin = Math.min(yMin, landmark.y);
    yMax = Math.max(yMax, landmark.y);

    detection.locationData.relativeKeypoints.push(
        {x: landmark.x, y: landmark.y} as Keypoint);
  }

  detection.locationData.relativeBoundingBox =
      {xMin, yMin, xMax, yMax, width: (xMax - xMin), height: (yMax - yMin)};

  return detection;
}
