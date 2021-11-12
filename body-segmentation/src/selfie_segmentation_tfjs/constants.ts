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

import {ImageToTensorConfig, TensorsToSegmentationConfig} from '../shared/calculators/interfaces/config_interfaces';
import {SelfieSegmentationTfjsModelConfig, SelfieSegmentationTfjsSegmentationConfig} from './types';

export const DEFAULT_SELFIESEGMENTATION_MODEL_URL_GENERAL =
    'https://storage.googleapis.com/tfjs-testing/selfie-segmentation/selfie-segmentation/model.json';
export const DEFAULT_SELFIESEGMENTATION_MODEL_URL_LANDSCAPE =
    'https://storage.googleapis.com/tfjs-testing/selfie-segmentation/selfie-segmentation-landscape/model.json';
export const DEFAULT_SELFIESEGMENTATION_MODEL_CONFIG:
    SelfieSegmentationTfjsModelConfig = {
      runtime: 'tfjs',
      modelType: 'general',
      modelUrl: DEFAULT_SELFIESEGMENTATION_MODEL_URL_GENERAL,
    };
export const DEFAULT_SELFIESEGMENTATION_SEGMENTATION_CONFIG:
    SelfieSegmentationTfjsSegmentationConfig = {
      flipHorizontal: false,
    };
export const SELFIESEGMENTATION_IMAGE_TO_TENSOR_GENERAL_CONFIG:
    ImageToTensorConfig = {
      inputResolution: {width: 256, height: 256},
      keepAspectRatio: false
    };
export const SELFIESEGMENTATION_IMAGE_TO_TENSOR_LANDSCAPE_CONFIG:
    ImageToTensorConfig = {
      inputResolution: {width: 256, height: 144},
      keepAspectRatio: false
    };
export const SELFIESEGMENTATION_TENSORS_TO_SEGMENTATION_CONFIG:
    TensorsToSegmentationConfig = {
      activation: 'none'
    };
