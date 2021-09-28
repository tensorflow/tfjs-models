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

import {Keypoint} from '../calculators/interfaces/common_interfaces';

import {VisibilitySmoothingConfig} from '../calculators/interfaces/config_interfaces';
import {LowPassFilter} from './low_pass_filter';

/**
 * Smoothing visibility using a `LowPassFilter` for each landmark.
 */
export class LowPassVisibilityFilter {
  private readonly alpha: number;
  private visibilityFilters: LowPassFilter[];

  constructor(config: VisibilitySmoothingConfig) {
    this.alpha = config.alpha;
  }

  apply(landmarks?: Keypoint[]): Keypoint[] {
    if (landmarks == null) {
      // Reset filters.
      this.visibilityFilters = null;
      return null;
    }
    if (this.visibilityFilters == null ||
        (this.visibilityFilters.length !== landmarks.length)) {
      // Initialize new filters.
      this.visibilityFilters =
          landmarks.map(_ => new LowPassFilter(this.alpha));
    }

    const outLandmarks = [];

    // Filter visibilities.
    for (let i = 0; i < landmarks.length; ++i) {
      const landmark = landmarks[i];

      const outLandmark = {...landmark};
      outLandmark.score = this.visibilityFilters[i].apply(landmark.score);

      outLandmarks.push(outLandmark);
    }

    return outLandmarks;
  }
}
