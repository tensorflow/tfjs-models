/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

/**
 * Rotates an image.
 *
 * @param image - Input tensor.
 * @param radians - Angle of rotation.
 * @param fillValue - The RGBA values to use in filling the leftover triangles
 * after rotation.
 * @param center - The center of rotation.
 */
export function rotate(
    image: tf.Tensor4D, radians: number, fillValue: number[]|number,
    center: [number, number]): tf.Tensor4D {
  const wasmBackend = tf.backend() as tfjsWasm.BackendWasm;

  const output = wasmBackend.makeOutput(image.shape, image.dtype);

  return tf.engine().makeTensorFromDataId(
             output.dataId, output.shape, output.dtype) as tf.Tensor4D;
}
