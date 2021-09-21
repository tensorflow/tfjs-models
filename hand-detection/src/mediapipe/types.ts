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

import {EstimationConfig, ModelConfig} from '../types';

/**
 * Common MediaPipeHands model config.
 */
export interface MediaPipeHandsModelConfig extends ModelConfig {
  runtime: 'mediapipe'|'tfjs';
}

export interface MediaPipeHandsEstimationConfig extends EstimationConfig {}

/**
 * Model parameters for MediaPipeHands MediaPipe runtime
 *
 * `runtime`: Must set to be 'mediapipe'.
 *
 * `solutionPath`: Optional. The path to where the wasm binary and model files
 * are located.
 */
export interface MediaPipeHandsMediaPipeModelConfig extends
    MediaPipeHandsModelConfig {
  runtime: 'mediapipe';
  solutionPath?: string;
}

/**
 * Hand estimation parameters for MediaPipeHands MediaPipe runtime.
 */
export interface MediaPipeHandsMediaPipeEstimationConfig extends
    MediaPipeHandsEstimationConfig {}
