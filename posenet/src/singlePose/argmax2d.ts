/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// does a floored division.  this is needed temporarily
// standard integer division can result in bugs
// https://github.com/PAIR-code/deeplearnjs/issues/847
function integerDiv(a: tf.Tensor1D, b: number) {
  const originalValues = a.buffer().values;
  const values = new Int32Array(a.shape[0]);

  for (let i = 0; i < a.shape[0]; i++) {
    values[i] = Math.floor(originalValues[i] / b);
  }

  return tf.tensor1d(values, 'int32');
}

function mod(a: tf.Tensor1D, b: number) {
  return tf.tidy(() => {
    const floored = integerDiv(a, b);

    return a.sub(floored.mul(tf.scalar(b, 'int32')));
  });
}

export function argmax2d(inputs: tf.Tensor3D): tf.Tensor2D {
  const [height, width, depth] = inputs.shape;

  return tf.tidy(() => {
    const reshaped = inputs.reshape([height * width, depth]);
    const coords = reshaped.argMax(0) as tf.Tensor1D;

    const yCoords = integerDiv(coords, width).expandDims(1) as tf.Tensor2D;
    const xCoords = mod(coords, width).expandDims(1) as tf.Tensor2D;

    return tf.concat([yCoords, xCoords], 1);
  })
}
