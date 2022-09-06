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

import {ModelConfig, SegmentationConfig} from '../types';

export type MediaPipeSelfieSegmentationModelType = 'general'|'landscape';

/**
 * Common SelfieSegmentation model config.
 */
export interface MediaPipeSelfieSegmentationModelConfig extends ModelConfig {
  runtime: 'mediapipe'|'tfjs';
  modelType?: MediaPipeSelfieSegmentationModelType;
}

export interface MediaPipeSelfieSegmentationSegmentationConfig extends
    SegmentationConfig {}

/**
 * Model parameters for SelfieSegmentation MediaPipe runtime
 *
 * `runtime`: Must set to be 'mediapipe'.
 *
 * `modelType`: Optional. Possible values: 'general'|'landscape'. Defaults to
 * 'general'.  The landscape model is similar to the general model, but operates
 * on a 144x256x3 tensor rather than 256x256x3. It has fewer FLOPs than the
 * general model, and therefore, runs faster
 *
 * `solutionPath`: Optional. The path to where the wasm binary and model files
 * are located.
 */
export interface MediaPipeSelfieSegmentationMediaPipeModelConfig extends
    MediaPipeSelfieSegmentationModelConfig {
  runtime: 'mediapipe';
  solutionPath?: string;
}

/**
 * Body segmentation parameters for SelfieSegmentation MediaPipe runtime.
 */
export interface MediaPipeSelfieSegmentationMediaPipeSegmentationConfig extends
    MediaPipeSelfieSegmentationSegmentationConfig {}
