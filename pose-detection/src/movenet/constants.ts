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

export const SINGLEPOSE_LIGHTNING = 'SinglePose.Lightning';
export const SINGLEPOSE_THUNDER = 'SinglePose.Thunder';
export const MULTIPOSE = 'MultiPose';

export const VALID_MODELS =
    [SINGLEPOSE_LIGHTNING, SINGLEPOSE_THUNDER, MULTIPOSE];

export const MOVENET_SINGLEPOSE_LIGHTNING_URL =
    'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
export const MOVENET_SINGLEPOSE_THUNDER_URL =
    'https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4';

export const MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION = 192;
export const MOVENET_SINGLEPOSE_THUNDER_RESOLUTION = 256;
export const MOVENET_MULTIPOSE_RESOLUTION = 320;

// The default configuration for loading MoveNet.
export const MOVENET_CONFIG: MoveNetModelConfig = {
  modelType: SINGLEPOSE_LIGHTNING,
  enableSmoothing: true
};

export const MOVENET_ESTIMATION_CONFIG: MoveNetEstimationConfig = {};

export const KEYPOINT_FILTER_CONFIG = {
  frequency: 30,
  minCutOff: 2.5,
  beta: 300.0,
  derivateCutOff: 2.5,
  thresholdCutOff: 0.5,
  thresholdBeta: 5.0
};
export const CROP_FILTER_ALPHA = 0.9;
export const MIN_CROP_KEYPOINT_SCORE = 0.2;
export const MIN_POSE_SCORE = 0.2;

export const NUM_KEYPOINTS = 17;
export const NUM_KEYPOINT_VALUES = 3;  // [y, x, score]
export const MULTIPOSE_BOX_SIZE = 5;   // [ymin, xmin, ymax, xmax, score]
export const MULTIPOSE_BOX_SCORE_IDX = NUM_KEYPOINTS * NUM_KEYPOINT_VALUES + 4;
export const MULTIPOSE_INSTANCE_SIZE =
    NUM_KEYPOINTS * NUM_KEYPOINT_VALUES + MULTIPOSE_BOX_SIZE;
