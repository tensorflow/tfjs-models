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

import {PoseNetEstimationConfig, PosenetModelConfig, PoseNetOutputStride} from './types';

// The default configuration for loading MobileNetV1 based PoseNet.
//
// (And for references, the default configuration for loading ResNet
// based PoseNet is also included).
//
// ```
// const RESNET_CONFIG = {
//   architecture: 'ResNet50',
//   outputStride: 32,
//   quantBytes: 2,
// } as ModelConfig;
// ```
export const MOBILENET_V1_CONFIG: PosenetModelConfig = {
  architecture: 'MobileNetV1',
  outputStride: 16,
  multiplier: 0.75,
  inputResolution: {height: 257, width: 257}
};

export const VALID_ARCHITECTURE = ['MobileNetV1', 'ResNet50'];
export const VALID_STRIDE = {
  'MobileNetV1': [8, 16],
  'ResNet50': [16]
};
export const VALID_OUTPUT_STRIDES: PoseNetOutputStride[] = [8, 16, 32];
export const VALID_MULTIPLIER = {
  'MobileNetV1': [0.50, 0.75, 1.0],
  'ResNet50': [1.0]
};
export const VALID_QUANT_BYTES = [1, 2, 4];

export const SINGLE_PERSON_ESTIMATION_CONFIG: PoseNetEstimationConfig = {
  maxPoses: 1,
  flipHorizontal: false
};

export const MULTI_PERSON_ESTIMATION_CONFIG: PoseNetEstimationConfig = {
  maxPoses: 5,
  flipHorizontal: false,
  scoreThreshold: 0.5,
  nmsRadius: 20
};

export const RESNET_MEAN = [-123.15, -115.90, -103.06];
