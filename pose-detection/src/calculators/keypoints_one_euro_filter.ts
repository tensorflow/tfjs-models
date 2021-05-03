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

import {KeypointsFilter} from './interfaces/common_interfaces';
import {OneEuroFilterConfig} from './interfaces/config_interfaces';
import {OneEuroFilter} from './one_euro_filter';

/**
 * A stateful filter that smoothes keypoints values overtime.
 *
 * More specifically, it uses `OneEuroFilter` to smooth every x, y, z
 * coordinates over time, which as result gives us velocity of how these values
 * change over time. With higher velocity it weights new values higher.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
export class KeypointsOneEuroFilter implements KeypointsFilter {
  private xFilters: OneEuroFilter[];
  private yFilters: OneEuroFilter[];
  private zFilters: OneEuroFilter[];

  constructor(private readonly config: OneEuroFilterConfig) {}

  apply(keypoints: Keypoint[], microSeconds: number): Keypoint[] {
    if (keypoints == null) {
      this.reset();
      return null;
    }

    // Initialize filters once.
    this.initializeFiltersIfEmpty(keypoints);

    // Filter keypoints. Every axis of every keypoint is filtered separately.
    return keypoints.map((keypoint, i) => {
      const outKeypoint = {
        ...keypoint,
        x: this.xFilters[i].apply(keypoint.x, microSeconds),
        y: this.yFilters[i].apply(keypoint.y, microSeconds),
      };

      if (keypoint.z != null) {
        outKeypoint.z = this.zFilters[i].apply(keypoint.z, microSeconds);
      }

      return outKeypoint;
    });
  }

  reset() {
    this.xFilters = null;
    this.yFilters = null;
    this.zFilters = null;
  }

  // Initializes filters for the first time or after reset. If initialized the
  // check the size.
  private initializeFiltersIfEmpty(keypoints: Keypoint[]) {
    if (this.xFilters == null || this.xFilters.length !== keypoints.length) {
      this.xFilters = keypoints.map(_ => new OneEuroFilter(this.config));
      this.yFilters = keypoints.map(_ => new OneEuroFilter(this.config));
      this.zFilters = keypoints.map(_ => new OneEuroFilter(this.config));
    }
  }
}
