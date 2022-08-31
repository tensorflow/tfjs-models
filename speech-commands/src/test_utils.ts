/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {Tensor, test_util, util} from '@tensorflow/tfjs-core';

export function expectTensorsClose(
    actual: Tensor|number[], expected: Tensor|number[], epsilon?: number) {
  if (actual == null) {
    throw new Error(
        'First argument to expectTensorsClose() is not defined.');
  }
  if (expected == null) {
    throw new Error(
        'Second argument to expectTensorsClose() is not defined.');
  }
  if (actual instanceof Tensor && expected instanceof Tensor) {
    if (actual.dtype !== expected.dtype) {
      throw new Error(
          `Data types do not match. Actual: '${actual.dtype}'. ` +
          `Expected: '${expected.dtype}'`);
    }
    if (!util.arraysEqual(actual.shape, expected.shape)) {
      throw new Error(
          `Shapes do not match. Actual: [${actual.shape}]. ` +
          `Expected: [${expected.shape}].`);
    }
  }
  const actualData = actual instanceof Tensor ? actual.dataSync() : actual;
  const expectedData =
      expected instanceof Tensor ? expected.dataSync() : expected;
  test_util.expectArraysClose(actualData, expectedData, epsilon);
}
