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

import {DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_CONFIG, DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_GENERAL, DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_LANDSCAPE, DEFAULT_TFJS_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG} from './constants';
import {MediaPipeSelfieSegmentationTfjsModelConfig, MediaPipeSelfieSegmentationTfjsSegmentationConfig} from './types';

export function validateModelConfig(
    modelConfig: MediaPipeSelfieSegmentationTfjsModelConfig):
    MediaPipeSelfieSegmentationTfjsModelConfig {
  if (modelConfig == null) {
    return {...DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_CONFIG};
  }

  const config: MediaPipeSelfieSegmentationTfjsModelConfig = {...modelConfig};

  config.runtime = 'tfjs';

  if (config.modelType == null) {
    config.modelType = DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_CONFIG.modelType;
  }

  if (config.modelType !== 'general' && config.modelType !== 'landscape') {
    throw new Error(`Model type must be one of general or landscape, but got ${
        config.modelType}`);
  }

  if (config.modelUrl == null) {
    switch (config.modelType) {
      case 'general':
        config.modelUrl = DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_GENERAL;
        break;
      case 'landscape':
      default:
        config.modelUrl = DEFAULT_TFJS_SELFIE_SEGMENTATION_MODEL_URL_LANDSCAPE;
        break;
    }
  }

  return config;
}

export function validateSegmentationConfig(
    segmentationConfig: MediaPipeSelfieSegmentationTfjsSegmentationConfig):
    MediaPipeSelfieSegmentationTfjsSegmentationConfig {
  if (segmentationConfig == null) {
    return {...DEFAULT_TFJS_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG};
  }

  const config = {...segmentationConfig};

  if (config.flipHorizontal == null) {
    config.flipHorizontal =
        DEFAULT_TFJS_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG.flipHorizontal;
  }

  return config;
}
