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

import {TrackerConfig} from './interfaces/config_interfaces';

export function validateTrackerConfig(config: TrackerConfig): void {
  if (config.maxTracks < 1) {
    throw new Error(
      `Must specify 'maxTracks' to be at least 1, but ` +
      `encountered ${config.maxTracks}`);
  }
  if (config.maxAge <= 0) {
    throw new Error(
      `Must specify 'maxAge' to be positive, but ` +
      `encountered ${config.maxAge}`);
  }

  if (config.keypointTrackerParams !== undefined) {
    if (config.keypointTrackerParams.keypointConfidenceThreshold < 0 ||
        config.keypointTrackerParams.keypointConfidenceThreshold > 1) {
      throw new Error(
        `Must specify 'keypointConfidenceThreshold' to be in the range ` +
        `[0, 1], but encountered ` +
        `${config.keypointTrackerParams.keypointConfidenceThreshold}`);
    }
    if (config.keypointTrackerParams.minNumberOfKeypoints < 1) {
      throw new Error(
        `Must specify 'minNumberOfKeypoints' to be at least 1, but ` +
        `encountered ${config.keypointTrackerParams.minNumberOfKeypoints}`);
    }
    for (const falloff of config.keypointTrackerParams.keypointFalloff) {
      if (falloff <= 0.0) {
        throw new Error(
          `Must specify each keypoint falloff parameterto be positive ` +
          `but encountered ${falloff}`);
      }
    }
  }
} 