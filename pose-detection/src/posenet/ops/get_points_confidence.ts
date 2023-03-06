/**
 * @license
 * Copyright 2023 Google LLC.
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

import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';

import {getPointsConfidenceWebGPU} from './get_points_confidence_webgpu';

export function getPointsConfidenceGPU<T extends tf.Tensor>(a: T, b: T): T {
  if (tf.backend() instanceof tfwebgpu.WebGPUBackend) {
    return getPointsConfidenceWebGPU(a, b);
  }

  throw new Error(
      'getPointsConfidenceWebGPU is not supported in this backend!');
}
