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

import { MOVENET_CONFIG } from './constants';
import { MoveNetEstimationConfig, MoveNetModelConfig } from './types';

export function validateModelConfig(modelConfig: MoveNetModelConfig):
  MoveNetModelConfig {
  const config = modelConfig || MOVENET_CONFIG;

  if (config.inputResolution == null) {
    config.inputResolution = { height: 192, width: 192 };
  }

  return config;
}

export function validateEstimationConfig(
  estimationConfig: MoveNetEstimationConfig): MoveNetEstimationConfig {
  const config = estimationConfig;

  if (config.maxPoses == null) {
    config.maxPoses = 1;
  }

  if (config.maxPoses <= 0) {
    throw new Error(`Invalid maxPoses ${config.maxPoses}. Should be > 0.`);
  }

  if (config.maxPoses > 1) {
    throw new Error(`Invalid maxPoses ${config.maxPoses}. > 1 not supported yet.`);
  }

  if (!config.minimumKeypointScore) {
    config.minimumKeypointScore = 0.3;
  }

  return config;
}
