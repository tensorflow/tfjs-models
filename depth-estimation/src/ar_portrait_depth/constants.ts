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
import {ARPortraitDepthModelConfig} from './types';

export const DEFAULT_AR_PORTRAIT_DEPTH_MODEL_URL =
    'https://tfhub.dev/tensorflow/tfjs-model/ar_portrait_depth/1';

export const DEFAULT_AR_PORTRAIT_DEPTH_MODEL_CONFIG:
    ARPortraitDepthModelConfig = {
      depthModelUrl: DEFAULT_AR_PORTRAIT_DEPTH_MODEL_URL,
    };

export const DEFAULT_AR_PORTRAIT_DEPTH_ESTIMATION_CONFIG = {
  flipHorizontal: false,
};
