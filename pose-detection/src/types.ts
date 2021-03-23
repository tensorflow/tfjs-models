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
  PoseNet = 'posenet',
  MediapipeBlazepose = 'mediapipe_blazepose'
}

export type QuantBytes = 1|2|4;

/**
 * Common config to create the pose detector.
 *
 * `quantBytes`: Optional. Options: 1, 2, or 4.  This parameter affects weight
 * quantization in the models. The available options are
 * 1 byte, 2 bytes, and 4 bytes. The higher the value, the larger the model size
 * and thus the longer the loading time, the lower the value, the shorter the
 * loading time but lower the accuracy.
 */
export interface ModelConfig {
  quantBytes?: QuantBytes;
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
  maxPoses: number;
  flipHorizontal?: boolean;
}

export interface InputResolution {
  width: number;
  height: number;
}

/**
 * A keypoint that contains coordinate information.
 */
export interface Keypoint {
  x: number;
  y: number;
  z?: number;
  score?: number;
  name?: string;
  visibility?: number;
}

export interface Pose {
  keypoints: Keypoint[];
  score?: number;
}
