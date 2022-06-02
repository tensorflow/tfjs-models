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

import {MediaPipeFaceMeshEstimationConfig, MediaPipeFaceMeshModelConfig} from '../mediapipe/types';

/**
 * Model parameters for MediaPipeFaceMesh TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `refineLandmarks`: Defaults to false. If set to true, refines the landmark
 * coordinates around the eyes and lips, and output additional landmarks around
 * the irises.
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 *
 * `landmarkModelUrl`: Optional. An optional string that specifies custom url of
 * the landmark model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
export interface MediaPipeFaceMeshTfjsModelConfig extends
    MediaPipeFaceMeshModelConfig {
  runtime: 'tfjs';
  detectorModelUrl?: string|io.IOHandler;
  landmarkModelUrl?: string|io.IOHandler;
}

/**
 * Face estimation parameters for MediaPipeFaceMesh TFJS runtime.
 */
export interface MediaPipeFaceMeshTfjsEstimationConfig extends
    MediaPipeFaceMeshEstimationConfig {}
