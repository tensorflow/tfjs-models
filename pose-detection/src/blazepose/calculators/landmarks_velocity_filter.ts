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
import {VelocityFilterConfig} from '../../calculators/interfaces/config_interfaces';
import {RelativeVelocityFilter} from '../../calculators/relative_velocity_filter';
import {getObjectScale} from '../../calculators/velocity_filter_utils';
import {Keypoint} from '../../types';
import {LandmarksFilter} from './interfaces/common_interfaces';

/**
 * A stateful filter that smoothes landmark values overtime.
 *
 * More specifically, it uses `RelativeVelocityFilter` to smooth every x, y, z
 * coordinates over time, which as result gives us velocity of how these values
 * change over time. With higher velocity it weights new values higher.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
export class LandmarksVelocityFilter implements LandmarksFilter {
  private xFilters: RelativeVelocityFilter[];
  private yFilters: RelativeVelocityFilter[];
  private zFilters: RelativeVelocityFilter[];

  constructor(private readonly config: VelocityFilterConfig) {}

  apply(landmarks: Keypoint[], microSeconds: number): Keypoint[] {
    if (landmarks == null) {
      this.reset();
      return null;
    }
    // Get value scale as inverse value of the object scale.
    // If value is too small smoothing will be disabled and landmarks will be
    // returned as is.
    let valueScale = 1;
    if (!this.config.disableValueScaling) {
      const objectScale = getObjectScale(landmarks);
      if (objectScale < this.config.minAllowedObjectScale) {
        return [...landmarks];
      }
      valueScale = 1 / objectScale;
    }

    // Initialize filters once.
    this.initializeFiltersIfEmpty(landmarks);

    // Filter landmarks. Every axis of every landmark is filtered separately.
    return landmarks.map((landmark, i) => {
      const outLandmark = {
        ...landmark,
        x: this.xFilters[i].apply(landmark.x, microSeconds, valueScale),
        y: this.yFilters[i].apply(landmark.y, microSeconds, valueScale),
      };

      if (landmark.z != null) {
        outLandmark.z =
            this.zFilters[i].apply(landmark.z, microSeconds, valueScale);
      }

      return outLandmark;
    });
  }

  reset() {
    this.xFilters = null;
    this.yFilters = null;
    this.zFilters = null;
  }

  // Initializes filters for the first time or after reset. If initialized the
  // check the size.
  private initializeFiltersIfEmpty(landmarks: Keypoint[]) {
    if (this.xFilters == null || this.xFilters.length !== landmarks.length) {
      this.xFilters =
          landmarks.map(_ => new RelativeVelocityFilter(this.config));
      this.yFilters =
          landmarks.map(_ => new RelativeVelocityFilter(this.config));
      this.zFilters =
          landmarks.map(_ => new RelativeVelocityFilter(this.config));
    }
  }
}
