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
import {ARPortraitDepthEstimationConfig} from './ar_portrait_depth/types';
import {DepthEstimatorInput, DepthMap} from './types';

/**
 * User-facing interface for all depth estimators.
 */
export interface DepthEstimator {
  /**
   * Estimates depth in the input image.
   *
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param estimationConfig common config for `estimateDepth`.
   */
  estimateDepth(
      input: DepthEstimatorInput,
      estimationConfig?: ARPortraitDepthEstimationConfig): Promise<DepthMap>;

  /**
   * Disposes the underlying models from memory.
   */
  dispose(): void;

  /**
   * Resets global states in the model.
   */
  reset(): void;
}
