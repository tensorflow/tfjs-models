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

export const ARCHITECTURE_SINGLEPOSE_LIGHTNING = 'SinglePose.Lightning';
export const ARCHITECTURE_SINGLEPOSE_THUNDER = 'SinglePose.Thunder';

export const VALID_ARCHITECTURES =
    [ARCHITECTURE_SINGLEPOSE_LIGHTNING, ARCHITECTURE_SINGLEPOSE_THUNDER];

export const MOVENET_SINGLEPOSE_LIGHTNING_URL =
    'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/1';
export const MOVENET_SINGLEPOSE_THUNDER_URL =
    'https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/1';

export const MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION = 192;
export const MOVENET_SINGLEPOSE_THUNDER_RESOLUTION = 256;

// The default configuration for loading MoveNet.
export const MOVENET_CONFIG: MoveNetModelConfig = {
  modelType: ARCHITECTURE_SINGLEPOSE_LIGHTNING
};

export const MOVENET_SINGLE_POSE_ESTIMATION_CONFIG: MoveNetEstimationConfig = {
  maxPoses: 1
};

export const MIN_CROP_KEYPOINT_SCORE = 0.3;

export const KPT_INDICES: {[index: string]: number} = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16
};