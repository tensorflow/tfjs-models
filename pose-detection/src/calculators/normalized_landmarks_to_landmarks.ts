import {Keypoint} from '..';
import {ImageSize} from './interfaces/common_interfaces';

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
export function normalizedLandmarksToLandmarks(
    normalizedLandmarks: Keypoint[], imageSize: ImageSize): Keypoint[] {
  return normalizedLandmarks.map(normalizedLandmark => {
    const landmark = {
      ...normalizedLandmark,
      x: normalizedLandmark.x * imageSize.width,
      y: normalizedLandmark.y * imageSize.height
    };

    if (normalizedLandmark.z != null) {
      // Scale z the same way as x (using image width).
      landmark.z = normalizedLandmark.z * imageSize.width;
    }

    return landmark;
  });
}