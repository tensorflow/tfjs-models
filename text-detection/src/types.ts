/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import {Point} from './geometry';

export interface TextDetectionConfig {
  quantizationBytes?: QuantizationBytes;
  modelUrl?: string;
}

export interface TextDetectionOptions {
  minKernelArea?: number;
  minScore?: number;
  maxSideLength?: number;
}
;

export type TextDetectionInput =
    |ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;

export type QuantizationBytes = 1|2|4;
export type Box = [Point, Point, Point, Point];
export type TextDetectionOutput = Box[];
