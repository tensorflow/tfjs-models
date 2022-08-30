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
import {MediaPipeSelfieSegmentationTfjsModelConfig, MediaPipeSelfieSegmentationTfjsSegmentationConfig} from './types';

export const DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_GENERAL =
    'https://tfhub.dev/mediapipe/tfjs-model/selfie_segmentation/general/1';
export const DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_LANDSCAPE =
    'https://tfhub.dev/mediapipe/tfjs-model/selfie_segmentation/landscape/1';
export const DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_CONFIG:
    MediaPipeSelfieSegmentationTfjsModelConfig = {
      runtime: 'tfjs',
      modelType: 'general',
      modelUrl: DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_GENERAL,
    };
export const DEFAULT_TFJS_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG:
    MediaPipeSelfieSegmentationTfjsSegmentationConfig = {
      flipHorizontal: false,
    };
export const SELFIE_SEGMENTATION_IMAGE_TO_TENSOR_GENERAL_CONFIG:
    ImageToTensorConfig = {
      outputTensorSize: {width: 256, height: 256},
      keepAspectRatio: false,
      borderMode: 'zero',
      outputTensorFloatRange: [0, 1]
    };
export const SELFIE_SEGMENTATION_IMAGE_TO_TENSOR_LANDSCAPE_CONFIG:
    ImageToTensorConfig = {
      outputTensorSize: {width: 256, height: 144},
      keepAspectRatio: false,
      borderMode: 'zero',
      outputTensorFloatRange: [0, 1]
    };
export const SELFIE_SEGMENTATION_TENSORS_TO_SEGMENTATION_CONFIG:
    TensorsToSegmentationConfig = {
      activation: 'none'
    };
