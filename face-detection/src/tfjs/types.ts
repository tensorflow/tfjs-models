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

import {MediaPipeFaceDetectorEstimationConfig, MediaPipeFaceDetectorModelConfig} from '../mediapipe/types';

/**
 * Model parameters for MediaPipeFaceDetector TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `modelType`: Optional. Possible values: 'short'|'full'. Defaults to
 * 'short'. The short-range model that works best for faces within 2 meters from
 * the camera, while the full-range model works best for faces within 5 meters.
 * For the full-range option, a sparse model is used for its improved inference
 * speed.
 *
 * `maxFaces`: Optional. Default to 1. The maximum number of faces that will
 * be detected by the model. The number of returned faces can be less than the
 * maximum (for example when no faces are present in the input).
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
export interface MediaPipeFaceDetectorTfjsModelConfig extends
    MediaPipeFaceDetectorModelConfig {
  runtime: 'tfjs';
  detectorModelUrl?: string|io.IOHandler;
}

/**
 * Face estimation parameters for MediaPipeFaceDetector TFJS runtime.
 */
export interface MediaPipeFaceDetectorTfjsEstimationConfig extends
    MediaPipeFaceDetectorEstimationConfig {}
