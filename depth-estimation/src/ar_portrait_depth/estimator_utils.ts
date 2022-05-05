/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {DEFAULT_AR_PORTRAIT_DEPTH_ESTIMATION_CONFIG, DEFAULT_AR_PORTRAIT_DEPTH_MODEL_URL} from './constants';
import {ARPortraitDepthEstimationConfig, ARPortraitDepthModelConfig} from './types';

export function validateModelConfig(modelConfig: ARPortraitDepthModelConfig):
    ARPortraitDepthModelConfig {
  if (modelConfig == null || modelConfig.minDepth == null ||
      modelConfig.maxDepth == null) {
    throw new Error(
        `A model config with minDepth and maxDepth set must be provided.`);
  }

  if (modelConfig.minDepth > modelConfig.maxDepth) {
    throw new Error('minDepth must be <= maxDepth.');
  }

  const config = {...modelConfig};

  if (config.depthModelUrl == null) {
    config.depthModelUrl = DEFAULT_AR_PORTRAIT_DEPTH_MODEL_URL;
  }

  return config;
}

export function validateEstimationConfig(
    estimationConfig: ARPortraitDepthEstimationConfig) {
  if (estimationConfig == null) {
    return {...DEFAULT_AR_PORTRAIT_DEPTH_ESTIMATION_CONFIG};
  }

  const config = {...estimationConfig};

  if (config.flipHorizontal == null) {
    config.flipHorizontal =
        DEFAULT_AR_PORTRAIT_DEPTH_ESTIMATION_CONFIG.flipHorizontal;
  }

  return config;
}
