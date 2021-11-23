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

import {DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_MODEL_CONFIG, DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG} from './constants';
import {MediaPipeSelfieSegmentationMediaPipeModelConfig, MediaPipeSelfieSegmentationMediaPipeSegmentationConfig} from './types';

export function validateModelConfig(
    modelConfig: MediaPipeSelfieSegmentationMediaPipeModelConfig):
    MediaPipeSelfieSegmentationMediaPipeModelConfig {
  if (modelConfig == null) {
    return {...DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_MODEL_CONFIG};
  }

  const config:
      MediaPipeSelfieSegmentationMediaPipeModelConfig = {...modelConfig};

  config.runtime = 'mediapipe';

  if (config.modelType == null) {
    config.modelType =
        DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_MODEL_CONFIG.modelType;
  }

  return config;
}

export function validateSegmentationConfig(
    segmentationConfig: MediaPipeSelfieSegmentationMediaPipeSegmentationConfig):
    MediaPipeSelfieSegmentationMediaPipeSegmentationConfig {
  if (segmentationConfig == null) {
    return {...DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG};
  }

  const config = {...segmentationConfig};

  if (config.flipHorizontal == null) {
    config.flipHorizontal =
        DEFAULT_MEDIAPIPE_SELFIE_SEGMENTATION_SEGMENTATION_CONFIG
            .flipHorizontal;
  }

  return config;
}
