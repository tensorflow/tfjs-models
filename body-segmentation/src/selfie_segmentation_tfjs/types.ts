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
import {MediaPipeSelfieSegmentationModelConfig, MediaPipeSelfieSegmentationSegmentationConfig} from '../selfie_segmentation_mediapipe/types';

/**
 * Model parameters for MediaPipeSelfieSegmentation TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `modelType`: Optional. Possible values: 'general'|'landscape'. Defaults to
 * 'general'.  The landscape model is similar to the general model, but operates
 * on a 144x256x3 tensor rather than 256x256x3. It has fewer FLOPs than the
 * general model, and therefore, runs faster
 *
 * `modelUrl`: Optional. An optional string that specifies custom url of
 * the segmentation model. This is useful for area/countries that don't have
 * access to the model hosted on tf.hub.
 *
 */
export interface MediaPipeSelfieSegmentationTfjsModelConfig extends
    MediaPipeSelfieSegmentationModelConfig {
  runtime: 'tfjs';
  modelUrl?: string|io.IOHandler;
}

/**
 * Body segmentation parameters for MediaPipeSelfieSegmentation TFJS runtime.
 */
export interface MediaPipeSelfieSegmentationTfjsSegmentationConfig extends
    MediaPipeSelfieSegmentationSegmentationConfig {}
