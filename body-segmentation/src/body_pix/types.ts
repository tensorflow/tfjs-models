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

import {ModelConfig as BodySegmentationModelConfig, SegmentationConfig} from '../types';
import {ModelConfig as BodyPixBaseModelConfig, MultiPersonInstanceInferenceConfig, PersonInferenceConfig} from './impl/body_pix_model';

/**
 * BodyPix model config.
 */
export interface BodyPixModelConfig extends BodyPixBaseModelConfig,
                                            BodySegmentationModelConfig {}

/**
 * Segmentation parameters for BodyPix.
 *
 * `multiSegmentation`: If set to true, then each person is segmented in a
 * separate output, otherwise all people are segmented together in one
 * segmentation.
 *
 * `segmentBodyParts`: If set to true, then 24 body parts are segmented in
 * the output, otherwise only foreground / background binary segmentation is
 * performed.
 */
export interface BodyPixSegmentationConfig extends
    SegmentationConfig, MultiPersonInstanceInferenceConfig,
    PersonInferenceConfig {
  multiSegmentation: boolean;
  segmentBodyParts: boolean;
}
