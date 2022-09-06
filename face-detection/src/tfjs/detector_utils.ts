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

import {DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE, DEFAULT_DETECTOR_MODEL_URL_SHORT, DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG, DEFAULT_FACE_DETECTOR_MODEL_CONFIG} from './constants';
import {MediaPipeFaceDetectorTfjsEstimationConfig, MediaPipeFaceDetectorTfjsModelConfig} from './types';

export function validateModelConfig(
    modelConfig: MediaPipeFaceDetectorTfjsModelConfig):
    MediaPipeFaceDetectorTfjsModelConfig {
  if (modelConfig == null) {
    return {...DEFAULT_FACE_DETECTOR_MODEL_CONFIG};
  }

  const config: MediaPipeFaceDetectorTfjsModelConfig = {...modelConfig};

  if (config.modelType == null) {
    config.modelType = DEFAULT_FACE_DETECTOR_MODEL_CONFIG.modelType;
  }

  if (config.maxFaces == null) {
    config.maxFaces = DEFAULT_FACE_DETECTOR_MODEL_CONFIG.maxFaces;
  }

  if (config.detectorModelUrl == null) {
    switch (config.modelType) {
      case 'full':
        config.detectorModelUrl = DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE;
        break;
      case 'short':
      default:
        config.detectorModelUrl = DEFAULT_DETECTOR_MODEL_URL_SHORT;
        break;
    }
  }

  return config;
}

export function validateEstimationConfig(
    estimationConfig: MediaPipeFaceDetectorTfjsEstimationConfig):
    MediaPipeFaceDetectorTfjsEstimationConfig {
  if (estimationConfig == null) {
    return {...DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG};
  }

  const config = {...estimationConfig};

  if (config.flipHorizontal == null) {
    config.flipHorizontal =
        DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG.flipHorizontal;
  }

  return config;
}
