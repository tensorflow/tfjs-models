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
 * Common MediaPipeFaceMesh model config.
 */
export interface MediaPipeFaceMeshModelConfig extends ModelConfig {
  runtime: 'mediapipe'|'tfjs';
  refineLandmarks: boolean;
}

export interface MediaPipeFaceMeshEstimationConfig extends EstimationConfig {}

/**
 * Model parameters for MediaPipeFaceMesh MediaPipe runtime
 *
 * `runtime`: Must set to be 'mediapipe'.
 *
 * `refineLandmarks`: If set to true, refines the landmark coordinates around
 * the eyes and lips, and output additional landmarks around the irises.
 *
 * `solutionPath`: Optional. The path to where the wasm binary and model files
 * are located.
 */
export interface MediaPipeFaceMeshMediaPipeModelConfig extends
    MediaPipeFaceMeshModelConfig {
  runtime: 'mediapipe';
  solutionPath?: string;
}

/**
 * Face estimation parameters for MediaPipeFaceMesh MediaPipe runtime.
 */
export interface MediaPipeFaceMeshMediaPipeEstimationConfig extends
    MediaPipeFaceMeshEstimationConfig {}
