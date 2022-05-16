/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {EstimationConfig, ModelConfig} from '../types';

/**
 * Model parameters for ARPortraitDepth.
 *
 * `segmentationModelUrl`: Optional. An optional string that specifies custom
 * url of the selfie segmentation model. This is useful for area/countries that
 * don't have access to the model hosted on tf.hub.
 *
 * `depthModelUrl`: Optional. An optional string that specifies custom url of
 * the portrait depth model. This is useful for area/countries that don't have
 * access to the model hosted on tf.hub.
 *
 */
export interface ARPortraitDepthModelConfig extends ModelConfig {
  segmentationModelUrl?: string|io.IOHandler;
  depthModelUrl?: string|io.IOHandler;
}

export interface ARPortraitDepthEstimationConfig extends EstimationConfig {}
