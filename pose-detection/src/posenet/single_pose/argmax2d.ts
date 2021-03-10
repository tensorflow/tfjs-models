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

function mod(a: tf.Tensor1D, b: number): tf.Tensor1D {
  return tf.tidy(() => {
    const floored = a.div(tf.scalar(b, 'int32'));

    return a.sub(floored.mul(tf.scalar(b, 'int32')));
  });
}

export function argmax2d(inputs: tf.Tensor3D): tf.Tensor2D {
  const [height, width, depth] = inputs.shape;

  return tf.tidy(() => {
    const reshaped = inputs.reshape([height * width, depth]);
    const coords = reshaped.argMax(0);

    const yCoords = coords.div(tf.scalar(width, 'int32')).expandDims(1);
    const xCoords = mod(coords as tf.Tensor1D, width).expandDims(1);

    return tf.concat([yCoords, xCoords], 1);
  }) as tf.Tensor2D;
}
