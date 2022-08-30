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
import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {arrayToMatrix4x4, calculateInverseMatrix, Matrix4x4, matrix4x4ToArray} from './calculate_inverse_matrix';

describe('calculateInverseMatrix', () => {
  const identity: Matrix4x4 =
      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

  it('identity matrix.', async () => {
    const inverse = calculateInverseMatrix(identity);

    expectArraysClose(matrix4x4ToArray(inverse), matrix4x4ToArray(identity));
  });

  it('translation.', async () => {
    const matrix: Matrix4x4 = [
      [1.0, 0.0, 0.0, 2.0],
      [0.0, 1.0, 0.0, -5.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    const inverse = calculateInverseMatrix(matrix);

    const expectedInverse: Matrix4x4 = [
      [1.0, 0.0, 0.0, -2.0],
      [0.0, 1.0, 0.0, 5.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    expectArraysClose(
        matrix4x4ToArray(inverse), matrix4x4ToArray(expectedInverse));
  });

  it('scale.', async () => {
    const matrix: Matrix4x4 = [
      [5.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    const inverse = calculateInverseMatrix(matrix);

    const expectedInverse: Matrix4x4 = [
      [0.2, 0.0, 0.0, 0.0],
      [0.0, 0.5, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    expectArraysClose(
        matrix4x4ToArray(inverse), matrix4x4ToArray(expectedInverse));
  });

  it('rotation90.', async () => {
    const matrix: Matrix4x4 = [
      [0.0, -1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    const inverse = calculateInverseMatrix(matrix);

    const expectedInverse: Matrix4x4 = [
      [0.0, 1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    expectArraysClose(
        matrix4x4ToArray(inverse), matrix4x4ToArray(expectedInverse));
  });

  it('precision.', async () => {
    const matrix: Matrix4x4 = [
      [0.00001, 0.0, 0.0, 0.0], [0.0, 0.00001, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ];

    const inverse = calculateInverseMatrix(matrix);

    const expectedInverse: Matrix4x4 = [
      [100000.0, 0.0, 0.0, 0.0], [0.0, 100000.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    ];

    expectArraysClose(
        matrix4x4ToArray(inverse), matrix4x4ToArray(expectedInverse));
  });

  it('random matrix.', async () => {
    for (let seed = 1; seed <= 5; ++seed) {
      const matrix = tf.randomUniform([4, 4], 0, 10, 'float32', seed);
      const inverse =
          calculateInverseMatrix(arrayToMatrix4x4(matrix.dataSync()));
      const product = tf.matMul(matrix, inverse);

      expectArraysClose(product.dataSync(), matrix4x4ToArray(identity));
    }
  });
});
