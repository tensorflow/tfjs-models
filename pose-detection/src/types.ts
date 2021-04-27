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
  MoveNet = 'MoveNet',
  MediapipeBlazepose = 'MediapipeBlazepose',
  PoseNet = 'PoseNet'
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
 * Common config for the `estimatePoses` method.
 *
 * `maxPoses`: Optional. Max number poses to detect. Default to 1, which means
 * single pose detection. Single pose detection runs more efficiently, while
 * multi-pose (maxPoses > 1) detection is usually much slower. Multi-pose
 * detection should only be used when needed.
 *
 * `flipHorizontal`: Optional. Default to false. In some cases, the image is
 * mirrored, e.g. video stream from camera, flipHorizontal will flip the
 * keypoints horizontally.
 */
export interface EstimationConfig {
  maxPoses?: number;
  flipHorizontal?: boolean;
}

/**
 * Allowed input format for the `estimatePoses` method.
 */
export type PoseDetectorInput =
    tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|HTMLCanvasElement;

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
  score?: number;  // The probability of a keypoint's visibility.
  name?: string;
}

export interface Pose {
  keypoints: Keypoint[];
  score?: number;  // The probability of an actual pose.
}
