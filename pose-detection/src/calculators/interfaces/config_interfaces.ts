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

import {InputResolution} from '../../types';

export interface ImageToTensorConfig {
  inputResolution: InputResolution;
  keepAspectRatio?: boolean;
}
export interface VelocityFilterConfig {
  windowSize?: number;  // Number of value changes to keep over time. Higher
                        // value adds to lag and to stability.
  velocityScale?:
      number;  // Scale to apply to the velocity calculated over the given
               // window. With higher velocity `low pass filter` weights new new
               // values higher. Lower value adds to lag and to stability.
  minAllowedObjectScale?:
      number;  // If calculated object scale is less than given value smoothing
               // will be disabled and landmarks will be returned as is.
  disableValueScaling?:
      boolean;  // Disable value scaling based on object size and use `1.0`
                // instead. Value scale is calculated as inverse value of object
                // size. Object size is calculated as maximum side of
                // rectangular bounding box of the object in XY plane.
}
export interface OneEuroFilterConfig {
  frequency?: number;  // Frequency of incoming frames defined in seconds. Used
                       // only if can't be calculated from provided events (e.g.
                       // on the very fist frame).
  minCutOff?:
      number;  // Minimum cutoff frequency. Start by tuning this parameter while
               // keeping `beta=0` to reduce jittering to the desired level. 1Hz
               // (the default value) is a a good starting point.
  beta?: number;  // Cutoff slope. After `minCutOff` is configured, start
                  // increasing `beta` value to reduce the lag introduced by the
                  // `minCutoff`. Find the desired balance between jittering and
                  // lag.
  derivateCutOff?:
      number;  // Cutoff frequency for derivate. It is set to 1Hz in the
               // original algorithm, but can be turned to further smooth the
               // speed (i.e. derivate) on the object.
  thresholdCutOff?:
      number;  // The outlier threshold offset, will lead to a generally more
               // reactive filter that will be less likely to discount outliers.
  thresholdBeta?: number;  // The outlier threshold slope, will lead to a filter
                           // that will more aggressively react whenever the
                           // keypoint speed increases by being less likely to
                           // consider that an observation is an outlier.
}
export interface KeypointsSmoothingConfig {
  velocityFilter?: VelocityFilterConfig;
  oneEuroFilter?: OneEuroFilterConfig;
}
