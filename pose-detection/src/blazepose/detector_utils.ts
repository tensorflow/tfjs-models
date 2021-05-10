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

import {DEFAULT_BLAZEPOSE_ESTIMATION_CONFIG, DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_FULL, DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_LITE, DEFAULT_BLAZEPOSE_MODEL_CONFIG} from './constants';
import {BlazePoseEstimationConfig, BlazePoseModelConfig} from './types';

export function validateModelConfig(modelConfig: BlazePoseModelConfig):
    BlazePoseModelConfig {
  const config = modelConfig == null ? {...DEFAULT_BLAZEPOSE_MODEL_CONFIG} :
                                       {...modelConfig};

  if (config.enableSmoothing == null) {
    config.enableSmoothing = DEFAULT_BLAZEPOSE_MODEL_CONFIG.enableSmoothing;
  }

  if (config.lite == null) {
    config.lite = DEFAULT_BLAZEPOSE_MODEL_CONFIG.lite;
  }

  if (config.detectorModelUrl == null) {
    config.detectorModelUrl = DEFAULT_BLAZEPOSE_MODEL_CONFIG.detectorModelUrl;
  }

  if (config.lite == null) {
    config.lite = false;
  }

  if (config.landmarkModelUrl == null) {
    config.landmarkModelUrl = config.lite ?
        DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_LITE :
        DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_FULL;
  }

  return config;
}

export function validateEstimationConfig(
    estimationConfig: BlazePoseEstimationConfig): BlazePoseEstimationConfig {
  let config;

  if (estimationConfig == null) {
    config = DEFAULT_BLAZEPOSE_ESTIMATION_CONFIG;
  } else {
    config = {...estimationConfig};
  }

  if (config.maxPoses == null) {
    config.maxPoses = 1;
  }

  if (config.maxPoses <= 0) {
    throw new Error(`Invalid maxPoses ${config.maxPoses}. Should be > 0.`);
  }

  if (config.maxPoses > 1) {
    throw new Error(
        'Multi-pose detection is not implemented yet. Please set maxPoses ' +
        'to 1.');
  }

  return config;
}
