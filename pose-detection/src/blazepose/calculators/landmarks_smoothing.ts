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

import {getImageSize} from '../../calculators/image_utils';
import {VelocityFilterConfig} from '../../calculators/interfaces/config_interfaces';
import {landmarksToNormalizedLandmarks} from '../../calculators/landmarks_to_normalized_landmarks';
import {LandmarksVelocityFilter} from '../../calculators/landmarks_velocity_filter';
import {normalizedLandmarksToLandmarks} from '../../calculators/normalized_landmarks_to_landmarks';
import {Keypoint} from '../../types';

/**
 * A Calculator to smooth normalized landmarks over time.
 */
export class LandmarksSmoothingFilter {
  private landmarksFilter: LandmarksVelocityFilter;

  constructor(private readonly config: VelocityFilterConfig) {
    this.landmarksFilter = new LandmarksVelocityFilter(config);
  }

  apply(landmarks: Keypoint[], image: HTMLVideoElement): Keypoint[] {
    if (landmarks == null) {
      this.landmarksFilter.reset();
      return null;
    }
    if (landmarks.length === 0) {
      this.landmarksFilter.reset();
      return null;
    }
    const imageSize = getImageSize(image);
    const macroSeconds = image.currentTime * 1e6;
    const scaledLandmarks =
        normalizedLandmarksToLandmarks(landmarks, imageSize);
    const scaledOutLandmarks =
        this.landmarksFilter.apply(scaledLandmarks, macroSeconds, this.config);

    return landmarksToNormalizedLandmarks(scaledOutLandmarks, imageSize);
  }
}
