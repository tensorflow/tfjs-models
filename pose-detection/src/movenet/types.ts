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
import {EstimationConfig, ModelConfig} from '../types';

/**
 * Additional MoveNet model loading config.
 *
 * 'modelType': Optional. The type of MoveNet model to load, Lighting or
 * Thunder. Defaults to Lightning. Lightning is a lower capacity model that can
 * run >50FPS on most modern laptops while achieving good performance. Thunder
 * is A higher capacity model that performs better prediction quality while
 * still achieving real-time (>30FPS) speed. Thunder will lag behind the
 * lightning, but it will pack a punch.
 *
 * `modelUrl`: Optional. An optional string that specifies custom url of the
 * model. This is useful for area/countries that don't have access to the model
 * hosted on TF Hub.
 */
export interface MoveNetModelConfig extends ModelConfig {
  modelType?: string;
  modelUrl?: string;
}

/**
 * MoveNet Specific Inference Config.
 *
 * `enableSmoothing`: Optional. Defaults to 'true'. When enabled, a temporal
 * smoothing filter will be used on the keypoint locations to reduce jitter.
 */
export interface MoveNetEstimationConfig extends EstimationConfig {
  enableSmoothing?: boolean;
}
