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
import {ImageSize, Keypoint} from './interfaces/common_interfaces';

export function keypointsToNormalizedKeypoints(
    keypoints: Keypoint[], imageSize: ImageSize): Keypoint[] {
  return keypoints.map(keypoint => {
    const normalizedKeypoint = {
      ...keypoint,
      x: keypoint.x / imageSize.width,
      y: keypoint.y / imageSize.height
    };

    if (keypoint.z != null) {
      // Scale z the same way as x (using image width).
      keypoint.z = keypoint.z / imageSize.width;
    }

    return normalizedKeypoint;
  });
}
