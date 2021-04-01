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
import {Keypoint} from '../types';
import {ImageSize} from './interfaces/common_interfaces';
export function landmarksToNormalizedLandmarks(
    landmarks: Keypoint[], imageSize: ImageSize): Keypoint[] {
  return landmarks.map(landmark => {
    const normalizedLandmark = {
      ...landmark,
      x: landmark.x / imageSize.width,
      y: landmark.y / imageSize.height
    };

    if (landmark.z != null) {
      // Scale z the same way as x (using image width).
      landmark.z = landmark.z / imageSize.width;
    }

    return normalizedLandmark;
  });
}
