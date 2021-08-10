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

import {Keypoint} from '../../types';

/**
 * A calculator to copy visibility between landmarks.
 *
 * Landmarks to copy from and to copy to can be of different type (normalized or
 * non-normalized), but landmarks to copy to and output landmarks should be of
 * the same type.
 * @param landmarksFrom  A LandmarkList or NormalizedLandmarklist of landmarks
 *     to copy from.
 * @param landmarksTo  A LandmarkList or NormalizedLandmarklist of landmarks
 *     to copy to.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/visibility_copy_calculator.cc
export function calculateVisibilityCopy(
    landmarksFrom: Keypoint[], landmarksTo: Keypoint[], copyVisibility = true) {
  const outputLandmarks = [];
  for (let i = 0; i < landmarksFrom.length; i++) {
    // Create output landmark and copy all fields from the `to` landmarks
    const newLandmark = {...landmarksTo[i]};

    // Copy visibility and presence from the `from` landmark.
    if (copyVisibility) {
      newLandmark.score = landmarksFrom[i].score;
    }

    outputLandmarks.push(newLandmark);
  }

  return outputLandmarks;
}
