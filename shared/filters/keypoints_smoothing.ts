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

import {getObjectScale} from '../calculators/get_object_scale';
import {ImageSize, Keypoint, KeypointsFilter} from '../calculators/interfaces/common_interfaces';
import {KeypointsSmoothingConfig} from '../calculators/interfaces/config_interfaces';
import {Rect} from '../calculators/interfaces/shape_interfaces';
import {keypointsToNormalizedKeypoints} from '../calculators/keypoints_to_normalized_keypoints';
import {normalizedKeypointsToKeypoints} from '../calculators/normalized_keypoints_to_keypoints';

import {KeypointsOneEuroFilter} from './keypoints_one_euro_filter';
import {KeypointsVelocityFilter} from './keypoints_velocity_filter';

/**
 * A Calculator to smooth keypoints over time.
 */
export class KeypointsSmoothingFilter {
  private readonly keypointsFilter: KeypointsFilter;

  constructor(config: KeypointsSmoothingConfig) {
    if (config.velocityFilter != null) {
      this.keypointsFilter = new KeypointsVelocityFilter(config.velocityFilter);
    } else if (config.oneEuroFilter != null) {
      this.keypointsFilter = new KeypointsOneEuroFilter(config.oneEuroFilter);
    } else {
      throw new Error(
          'Either configure velocityFilter or oneEuroFilter, but got ' +
          `${config}.`);
    }
  }

  /**
   * Apply one of the stateful `KeypointsFilter` to keypoints.
   *
   * Currently supports `OneEuroFilter` and `VelocityFilter`.
   * @param keypoints A list of 2D or 3D keypoints, can be normalized or
   *     non-normalized.
   * @param timestamp The timestamp of the video frame.
   * @param imageSize Optional. The imageSize is useful when keypoints are
   *     normalized.
   * @param normalized Optional. Whether the keypoints are normalized. Default
   *     to false.
   * @param objectScaleROI Optional. The auxiliary ROI to calculate object
   *     scale. If not set, objectScale defaults to 1.
   */
  apply(
      keypoints: Keypoint[], timestamp: number, imageSize?: ImageSize,
      normalized = false, objectScaleROI?: Rect): Keypoint[] {
    if (keypoints == null) {
      this.keypointsFilter.reset();
      return null;
    }

    const objectScale =
        objectScaleROI != null ? getObjectScale(objectScaleROI, imageSize) : 1;

    const scaledKeypoints = normalized ?
        normalizedKeypointsToKeypoints(keypoints, imageSize) :
        keypoints;
    const scaledKeypointsFiltered =
        this.keypointsFilter.apply(scaledKeypoints, timestamp, objectScale);

    return normalized ?
        keypointsToNormalizedKeypoints(scaledKeypointsFiltered, imageSize) :
        scaledKeypointsFiltered;
  }
}
