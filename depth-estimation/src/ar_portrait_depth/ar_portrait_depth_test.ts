/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as depthEstimation from '../index';
import {toImageDataLossy} from '../shared/calculators/mask_util';
import {loadImage} from '../shared/test_util';

// Measured in image channels (values between 0 and 255).
const EPSILON_IMAGE = 1;

function step(edge: number, x: number) {
  return x < edge ? 0 : 1;
}

function mix(
    x: [number, number, number], y: [number, number, number], a: number) {
  return x.map((value, i) => value * (1 - a) + y[i] * a);
}

function saturate(x: number) {
  return Math.max(Math.min(x, 1), 0);
}

function turboPlus(x: number) {
  return tf.tidy(() => {
    const d = 1. / 32.;
    const COLORS: Array<[number, number, number]> = [
      [0.4796, 0.0158, 0.0106],
      [0.6754, 0.0898, 0.0045],
      [0.8240, 0.1918, 0.0197],
      [0.9262, 0.3247, 0.0584],
      [0.9859, 0.5048, 0.1337],
      [0.9916, 0.6841, 0.2071],
      [0.9267, 0.8203, 0.2257],
      [0.7952, 0.9303, 0.2039],
      [0.6332, 0.9919, 0.2394],
      [0.4123, 0.9927, 0.3983],
      [0.1849, 0.9448, 0.6071],
      [0.0929, 0.8588, 0.7724],
      [0.1653, 0.7262, 0.9316],
      [0.2625, 0.5697, 0.9977],
      [0.337, 0.443, 0.925],
      [0.365, 0.306, 0.859],
      [0.4310, 0.1800, 0.827],
      [0.576, 0.118, 0.859],
      [0.737, 0.200, 0.886],
      [0.8947, 0.2510, 0.9137],
      [1.0000, 0.3804, 0.8431],
      [1.0000, 0.4902, 0.7451],
      [1.0000, 0.5961, 0.6471],
      [1.0000, 0.6902, 0.6039],
      [1.0000, 0.7333, 0.6157],
      [1.0000, 0.7804, 0.6431],
      [1.0000, 0.8275, 0.6824],
      [1.0000, 0.8706, 0.7255],
      [1.0000, 0.9098, 0.7765],
      [1.0000, 0.9451, 0.8235],
      [1.0000, 0.9725, 0.8588],
      [1.0000, 0.9922, 0.8863],
      [1., 1., 1.]
    ];

    let col = [0, 0, 0];
    for (let i = 0.; i < 31.; i++) {
      const scale = step(d * i, x) - step(d * (i + 1.), x);
      const RHS = mix(COLORS[i], COLORS[i + 1], saturate((x - d * i) / d))
                      .map(value => scale * value);
      col = col.map((value, i) => value + RHS[i]);
    }
    // Adds the last white colors after 99%.
    const scale = step(.99, x);
    const RHS = mix(COLORS[31], COLORS[32], saturate((x - .99) / .01))
                    .map(value => scale * value);
    col = col.map((value, i) => value + RHS[i]);

    return col.map(value => value * 255);
  });
}

describeWithFlags('ARPortraitDepth', ALL_ENVS, () => {
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('estimateDepth does not leak memory.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const estimator = await depthEstimation.createEstimator(
        depthEstimation.SupportedModels.ARPortraitDepth);
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    const depthMap =
        await estimator.estimateDepth(input, {minDepth: 0, maxDepth: 1});

    (await depthMap.toTensor()).dispose();
    expect(tf.memory().numTensors).toEqual(beforeTensors);

    estimator.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });

  it('throws error when minDepth is not set.', async (done) => {
    try {
      const estimator = await depthEstimation.createEstimator(
          depthEstimation.SupportedModels.ARPortraitDepth);
      const input: tf.Tensor3D = tf.zeros([128, 128, 3]);
      await estimator.estimateDepth(input);
      done.fail('Loading without minDepth succeeded unexpectedly.');
    } catch (e) {
      expect(e.message).toEqual(
          'An estimation config with ' +
          'minDepth and maxDepth set must be provided.');
      done();
    }
  });

  it('throws error when minDepth is greater than maxDepth.', async (done) => {
    try {
      const estimator = await depthEstimation.createEstimator(
          depthEstimation.SupportedModels.ARPortraitDepth);
      const input: tf.Tensor3D = tf.zeros([128, 128, 3]);
      await estimator.estimateDepth(input, {minDepth: 1, maxDepth: 0.99});
      done.fail(
          'Loading with minDepth greater than maxDepth ' +
          'succeeded unexpectedly.');
    } catch (e) {
      expect(e.message).toEqual('minDepth must be <= maxDepth.');
      done();
    }
  });
});

describeWithFlags('ARPortraitDepth static image ', BROWSER_ENVS, () => {
  let estimator: depthEstimation.DepthEstimator;
  let image: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('portrait.jpg', 192, 256);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function testBackend(backendName: 'cpu'|'webgl') {
    // Get expected depth values.
    const expectedDepthImage = await loadImage('depth.png', 192, 256);
    const expectedDepthValuesRGBA = await toImageDataLossy(expectedDepthImage);
    // Remove alpha channel.
    const expectedDepthValuesRGB =
        expectedDepthValuesRGBA.data.filter((_, i) => i % 4 !== 3);

    tf.setBackend(backendName);
    const startTensors = tf.memory().numTensors;

    // Get actual depth values.
    // Note: this makes a network request for model assets.
    estimator = await depthEstimation.createEstimator(
        depthEstimation.SupportedModels.ARPortraitDepth);

    const beforeTensors = tf.memory().numTensors;

    const result =
        await estimator.estimateDepth(image, {minDepth: 0.2, maxDepth: 0.9});
    const actualDepthValues = await result.toTensor();
    const coloredDepthValues =
        actualDepthValues.arraySync().flat().map(value => turboPlus(value));

    expectArraysClose(
        coloredDepthValues, expectedDepthValuesRGB, EPSILON_IMAGE);

    actualDepthValues.dispose();

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    estimator.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  }

  it('test CPU.', async () => {
    await testBackend('cpu');
  });

  it('test WebGL.', async () => {
    await testBackend('webgl');
  });
});
