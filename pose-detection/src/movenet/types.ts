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
 * MoveNet model loading config.
 *
 * `enableSmoothing`: Optional. A boolean indicating whether to use temporal
 * filter to smooth the predicted keypoints. Defaults to True. The temporal
 * filter relies on the currentTime field of the HTMLVideoElement. You can
 * override this timestamp by passing in your own timestamp (in milliseconds)
 * as the third parameter in `estimatePoses`. This is useful when the input is
 * a tensor, which doesn't have the currentTime field. Or in testing, to
 * simulate different FPS.
 *
 * `modelType`: Optional. The type of MoveNet model to load, Lighting or
 * Thunder. Defaults to Lightning. Lightning is a lower capacity model that can
 * run >50FPS on most modern laptops while achieving good performance. Thunder
 * is a higher capacity model that performs better prediction quality while
 * still achieving real-time (>30FPS) speed. Thunder will lag behind the
 * lightning, but it will pack a punch.
 *
 * `modelUrl`: Optional. An optional string that specifies custom url of the
 * model. This is useful for area/countries that don't have access to the model
 * hosted on TF Hub. If not provided, it will load the model specified by
 * `modelType` from tf.hub.
 */
export interface MoveNetModelConfig extends ModelConfig {
  enableSmoothing?: boolean;
  modelType?: string;
  modelUrl?: string;
}

/**
 * MoveNet Specific Inference Config.
 *
 * `maxPoses`: Optional. Defaults to 1. MoveNet only supports 1 pose for now.
 */
export interface MoveNetEstimationConfig extends EstimationConfig {}
