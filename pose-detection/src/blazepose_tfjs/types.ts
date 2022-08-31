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

import {io} from '@tensorflow/tfjs-core';

import {BlazePoseEstimationConfig, BlazePoseModelConfig} from '../blazepose_mediapipe/types';

/**
 * Model parameters for BlazePose TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `enableSmoothing`: Optional. A boolean indicating whether to use temporal
 * filter to smooth the predicted keypoints. Defaults to True. The temporal
 * filter relies on the currentTime field of the HTMLVideoElement. You can
 * override this timestamp by passing in your own timestamp (in milliseconds)
 * as the third parameter in `estimatePoses`. This is useful when the input is
 * a tensor, which doesn't have the currentTime field. Or in testing, to
 * simulate different FPS.
 *
 * `modelType`: Optional. Possible values: 'lite'|'full'|'heavy'. Defaults to
 * 'full'. The model accuracy increases from lite to heavy, while the inference
 * speed decreases and memory footprint increases. The heavy variant is intended
 * for applications that require high accuracy, while the lite variant is
 * intended for latency-critical applications. The full variant is a balanced
 * option.
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 *
 * `landmarkModelUrl`: Optional. An optional string that specifies custom url of
 * the landmark model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
export interface BlazePoseTfjsModelConfig extends BlazePoseModelConfig {
  runtime: 'tfjs';
  detectorModelUrl?: string|io.IOHandler;
  landmarkModelUrl?: string|io.IOHandler;
}

/**
 * Pose estimation parameters for BlazePose TFJS runtime.
 *
 * `maxPoses`: Optional. Defaults to 1. BlazePose only supports 1 pose for now.
 *
 * `flipHorizontal`: Optional. Default to false. When image data comes from
 * camera, the result has to flip horizontally.
 */
export interface BlazePoseTfjsEstimationConfig extends
    BlazePoseEstimationConfig {}
