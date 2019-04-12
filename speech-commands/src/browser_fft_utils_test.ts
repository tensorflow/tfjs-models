/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';
import {test_util} from '@tensorflow/tfjs';
import {normalize, normalizeFloat32Array} from './browser_fft_utils';

describe('normalize', () => {
  it('Non-constant value; no memory leak', () => {
    const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const numTensors0 = tf.memory().numTensors;
    const y = normalize(x);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0 + 1);
    test_util.expectArraysClose(
        y,
        tf.tensor4d(
            [-1.3416406, -0.4472135, 0.4472135, 1.3416406], [1, 2, 2, 1]));
    const {mean, variance} = tf.moments(y);
    test_util.expectArraysClose(mean, tf.scalar(0));
    test_util.expectArraysClose(variance, tf.scalar(1));
  });

  it('Constant value', () => {
    const x = tf.tensor4d([42, 42, 42, 42], [1, 2, 2, 1]);
    const y = normalize(x);
    test_util.expectArraysClose(y, tf.tensor4d([0, 0, 0, 0], [1, 2, 2, 1]));
  });
});

describe('normalizeFloat32Array', () => {
  it('Length-4 input', () => {
    const xs = new Float32Array([1, 2, 3, 4]);
    const numTensors0 = tf.memory().numTensors;
    const ys = normalizeFloat32Array(xs);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
    test_util.expectArraysClose(
        ys, new Float32Array([-1.3416406, -0.4472135, 0.4472135, 1.3416406]));
  });
});