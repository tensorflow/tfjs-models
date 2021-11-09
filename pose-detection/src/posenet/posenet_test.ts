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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as poseDetection from '../index';

describeWithFlags('PoseNet', ALL_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    // This test suite makes real network request for model assets, increase
    // the default timeout to allow enough time to load and reduce flakiness.
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 300000;  // 5mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  beforeEach(async () => {
    // Note: this makes a network request for model assets.
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.PoseNet, {
          quantBytes: 4,
          architecture: 'MobileNetV1',
          outputStride: 16,
          inputResolution: {width: 514, height: 513},
          multiplier: 1
        });
  });

  it('estimatePoses does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimatePoses(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('estimatePoses with multiple poses does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimatePoses(input, {maxPoses: 2});

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });
});
