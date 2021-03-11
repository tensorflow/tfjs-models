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
import * as tf from '@tensorflow/tfjs-core';

export enum SupportedModels {
  posenet = 'posenet'
}

/**
 * Allowed input format for the `estimatePoses` method.
 */
export type PoseDetectorInput =
    tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|HTMLCanvasElement;

/**
 * Common config for the `estimatePoses` method.
 */
export interface EstimationConfig {
  maxPoses?: number;
  flipHorizontal?: boolean;
  smoothLandmarks?: boolean;
  maxContinousChecks?: number;
}

/**
 * A keypoint that contains coordinate information.
 */
export interface Keypoint {
  x: number, y: number
}
