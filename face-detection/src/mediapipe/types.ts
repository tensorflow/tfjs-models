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

export type MediaPipeFaceDetectorModelType = 'short'|'full';

/**
 * Common MediaPipeFaceDetector model config.
 */
export interface MediaPipeFaceDetectorModelConfig extends ModelConfig {
  modelType?: MediaPipeFaceDetectorModelType;
  runtime: 'mediapipe'|'tfjs';
}

export interface MediaPipeFaceDetectorEstimationConfig extends
    EstimationConfig {}

/**
 * Model parameters for MediaPipeFaceDetector MediaPipe runtime
 *
 * `modelType`: Optional. Possible values: 'short'|'full'. Defaults to
 * 'short'. The short-range model that works best for faces within 2 meters from
 * the camera, while the full-range model works best for faces within 5 meters.
 * For the full-range option, a sparse model is used for its improved inference
 * speed.
 *
 * `runtime`: Must set to be 'mediapipe'.
 *
 * `solutionPath`: Optional. The path to where the wasm binary and model files
 * are located.
 */
export interface MediaPipeFaceDetectorMediaPipeModelConfig extends
    MediaPipeFaceDetectorModelConfig {
  runtime: 'mediapipe';
  solutionPath?: string;
}

/**
 * Face estimation parameters for MediaPipeFaceDetector MediaPipe runtime.
 */
export interface MediaPipeFaceDetectorMediaPipeEstimationConfig extends
    MediaPipeFaceDetectorEstimationConfig {}
