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

import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

export const VALID_ARCHITECTURES =
    ['SinglePose.Lightning', 'SinglePose.Thunder'];

export const MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION = 192;
export const MOVENET_SINGLEPOSE_THUNDER_RESOLUTION = 256;

// The default configuration for loading MoveNet.
export const MOVENET_CONFIG: MoveNetModelConfig = {
  modelType: 'SinglePose.Lightning'
};

export const MOVENET_SINGLE_POSE_ESTIMATION_CONFIG: MoveNetEstimationConfig = {
  maxPoses: 1,
  minimumKeypointScore: 0.3
};
