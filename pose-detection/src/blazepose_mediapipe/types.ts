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

export type BlazePoseModelType = 'lite'|'full'|'heavy';

/**
 * Common BlazePose model config.
 */
export interface BlazePoseModelConfig extends ModelConfig {
  runtime: 'mediapipe'|'tfjs';
  enableSmoothing?: boolean;
  enableSegmentation?: boolean;
  smoothSegmentation?: boolean;
  modelType?: BlazePoseModelType;
}

export interface BlazePoseEstimationConfig extends EstimationConfig {}

/**
 * Model parameters for BlazePose MediaPipe runtime
 *
 * `runtime`: Must set to be 'mediapipe'.
 *
 * `enableSmoothing`: Optional. A boolean indicating whether to use temporal
 * filter to smooth the predicted keypoints. Defaults to True. The temporal
 * filter relies on `performance.now()`. You can override this timestamp by
 * passing in your own timestamp (in milliseconds) as the third parameter in
 * `estimatePoses`.
 *
 * `enableSegmentation`: Optional. A boolean indicating whether to generate the
 * segmentation mask. Defaults to false.
 *
 * `smoothSegmentation`: Optional. A boolean indicating whether the solution
 * filters segmentation masks across different input images to reduce jitter.
 * Ignored if `enableSegmentation` is false or static images are passed in.
 * Defaults to true.
 *
 * `modelType`: Optional. Possible values: 'lite'|'full'|'heavy'. Defaults to
 * 'full'. The model accuracy increases from lite to heavy, while the inference
 * speed decreases and memory footprint increases. The heavy variant is intended
 * for applications that require high accuracy, while the lite variant is
 * intended for latency-critical applications. The full variant is a balanced
 * option.
 *
 * `solutionPath`: Optional. The path to where the wasm binary and model files
 * are located.
 */
export interface BlazePoseMediaPipeModelConfig extends BlazePoseModelConfig {
  runtime: 'mediapipe';
  solutionPath?: string;
}

/**
 * Pose estimation parameters for BlazePose MediaPipe runtime.
 *
 * `maxPoses`: Optional. Defaults to 1. BlazePose only supports 1 pose for now.
 *
 * `flipHorizontal`: Optional. Default to false. When image data comes from
 * camera, the result has to flip horizontally.
 */
export interface BlazePoseMediaPipeEstimationConfig extends
    BlazePoseEstimationConfig {}
