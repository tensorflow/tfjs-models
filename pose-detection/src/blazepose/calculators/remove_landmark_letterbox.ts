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

import {Padding} from '../../calculators/interfaces/common_interfaces';
import {Keypoint} from '../../types';

/**
 * Adjusts landmark locations on a letterboxed image to the corresponding
 * locations on the same image with the letterbox removed.
 * @param rawLandmark A NormalizedLandmarkList representing landmarks on an
 * letterboxed image.
 * @param padding A `padding` representing the letterbox padding from the 4
 *     sides, left, top, right, bottom, of the letterboxed image, normalized by
 *     the letterboxed image dimensions.
 * @returns Normalized landmarks.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmark_letterbox_removal_calculator.cc
export function removeLandmarkLetterbox(
    rawLandmark: Keypoint[], padding: Padding) {
  const left = padding.left;
  const top = padding.top;
  const leftAndRight = padding.left + padding.right;
  const topAndBottom = padding.top + padding.bottom;

  for (let i = 0; i < rawLandmark.length; ++i) {
    const landmark = rawLandmark[i];
    const newX = (landmark.x - left) / (1 - leftAndRight);
    const newY = (landmark.y - top) / (1 - topAndBottom);
    const newZ =
        (landmark.z - top) / (1 - leftAndRight);  // Scale Z coordinate as X.
    landmark.x = newX;
    landmark.y = newY;
    landmark.z = newZ;
  }

  return rawLandmark;
}
