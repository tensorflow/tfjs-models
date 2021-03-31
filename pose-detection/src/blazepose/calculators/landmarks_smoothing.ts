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

import {SECOND_TO_MICRO_SECONDS} from '../../calculators/constants';
import {getImageSize} from '../../calculators/image_utils';
import {landmarksToNormalizedLandmarks} from '../../calculators/landmarks_to_normalized_landmarks';
import {normalizedLandmarksToLandmarks} from '../../calculators/normalized_landmarks_to_landmarks';
import {Keypoint} from '../../types';
import {LandmarksFilter} from './interfaces/common_interfaces';
import {LandmarksSmoothingConfig} from './interfaces/config_interfaces';
import {LandmarksOneEuroFilter} from './landmarks_one_euro_filter';
import {LandmarksVelocityFilter} from './landmarks_velocity_filter';

/**
 * A Calculator to smooth normalized landmarks over time.
 */
export class LandmarksSmoothingFilter {
  private landmarksFilter: LandmarksFilter;

  constructor(config: LandmarksSmoothingConfig) {
    if (config.velocityFilter != null) {
      this.landmarksFilter = new LandmarksVelocityFilter(config.velocityFilter);
    } else if (config.oneEuroFilter != null) {
      this.landmarksFilter = new LandmarksOneEuroFilter(config.oneEuroFilter);
    } else {
      throw new Error(
          'Either configure velocityFilter or oneEuroFilter, but got ' +
          `${config}.`);
    }
  }

  apply(landmarks: Keypoint[], image: HTMLVideoElement): Keypoint[] {
    if (landmarks == null) {
      this.landmarksFilter.reset();
      return null;
    }
    const imageSize = getImageSize(image);
    const microSeconds = image.currentTime * SECOND_TO_MICRO_SECONDS;
    const scaledLandmarks =
        normalizedLandmarksToLandmarks(landmarks, imageSize);
    const scaledOutLandmarks =
        this.landmarksFilter.apply(scaledLandmarks, microSeconds);

    return landmarksToNormalizedLandmarks(scaledOutLandmarks, imageSize);
  }
}
