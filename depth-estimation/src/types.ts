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
import {Tensor2D} from '@tensorflow/tfjs-core';

import {PixelInput} from './shared/calculators/interfaces/common_interfaces';

export enum SupportedModels {
  ARPortraitDepth = 'ARPortraitDepth',
}

/**
 * Common config to create the depth estimator.
 */
export interface ModelConfig {}

/**
 * Common config for the `estimateDepth` method.
 *
 * `minDepth`: The minimum depth value for the model to map to 0. Any smaller
 * depth values will also get mapped to 0.
 *
 * `maxDepth`: The maximum depth value for the model to map to 1. Any larger
 * depth values will also get mapped to 1.
 *
 * `flipHorizontal`: Optional. Default to false. In some cases, the image is
 * mirrored, e.g., video stream from camera, flipHorizontal will flip the
 * keypoints horizontally.
 */
export interface EstimationConfig {
  minDepth: number;
  maxDepth: number;
  flipHorizontal?: boolean;
}

/**
 * Allowed input format for the `estimateDepth` method.
 */
export type DepthEstimatorInput = PixelInput;

export interface DepthMap {
  toCanvasImageSource(): Promise<CanvasImageSource>;

  toArray(): Promise<number[][]>;

  toTensor(): Promise<Tensor2D>;

  getUnderlyingType(): 'canvasimagesource'|'array'|
      'tensor'; /* determines which type the map currently stores in its
                   implementation so that conversion can be avoided */
}
