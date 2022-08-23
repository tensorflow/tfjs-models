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

export function concatWithNulls(
    ndarray1: tf.Tensor2D, ndarray2: tf.Tensor2D): tf.Tensor2D {
  if (ndarray1 == null && ndarray2 == null) {
    return null;
  }
  if (ndarray1 == null) {
    return ndarray2.clone();
  } else if (ndarray2 === null) {
    return ndarray1.clone();
  }
  return tf.concat<tf.Tensor2D>([ndarray1, ndarray2], 0);
}

export function topK(values: Float32Array, k: number):
    {values: Float32Array, indices: Int32Array} {
  const valuesAndIndices: Array<{value: number, index: number}> = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(k);
  const topkIndices = new Int32Array(k);
  for (let i = 0; i < k; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }
  return {values: topkValues, indices: topkIndices};
}
