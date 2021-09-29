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
import {Keypoint, KeypointsFilter} from '../calculators/interfaces/common_interfaces';
import {VelocityFilterConfig} from '../calculators/interfaces/config_interfaces';
import {RelativeVelocityFilter} from './relative_velocity_filter';

/**
 * A stateful filter that smoothes landmark values overtime.
 *
 * More specifically, it uses `RelativeVelocityFilter` to smooth every x, y, z
 * coordinates over time, which as result gives us velocity of how these values
 * change over time. With higher velocity it weights new values higher.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
export class KeypointsVelocityFilter implements KeypointsFilter {
  private xFilters: RelativeVelocityFilter[];
  private yFilters: RelativeVelocityFilter[];
  private zFilters: RelativeVelocityFilter[];

  constructor(private readonly config: VelocityFilterConfig) {}

  apply(keypoints: Keypoint[], microSeconds: number, objectScale: number):
      Keypoint[] {
    if (keypoints == null) {
      this.reset();
      return null;
    }
    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and keypoints will be
    // returned as is.
    let valueScale = 1;
    if (!this.config.disableValueScaling) {
      if (objectScale < this.config.minAllowedObjectScale) {
        return [...keypoints];
      }
      valueScale = 1 / objectScale;
    }

    // Initialize filters once.
    this.initializeFiltersIfEmpty(keypoints);

    // Filter keypoints. Every axis of every keypoint is filtered separately.
    return keypoints.map((keypoint, i) => {
      const outKeypoint = {
        ...keypoint,
        x: this.xFilters[i].apply(keypoint.x, microSeconds, valueScale),
        y: this.yFilters[i].apply(keypoint.y, microSeconds, valueScale),
      };

      if (keypoint.z != null) {
        outKeypoint.z =
            this.zFilters[i].apply(keypoint.z, microSeconds, valueScale);
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
      this.xFilters =
          keypoints.map(_ => new RelativeVelocityFilter(this.config));
      this.yFilters =
          keypoints.map(_ => new RelativeVelocityFilter(this.config));
      this.zFilters =
          keypoints.map(_ => new RelativeVelocityFilter(this.config));
    }
  }
}
