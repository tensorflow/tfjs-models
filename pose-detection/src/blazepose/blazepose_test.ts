/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as poseDetection from '../index';
import {loadImage} from '../test_util';

const modelType = ['fullbody', 'upperbody'];

const EXPECTED_UPPERBODY_LANDMARKS = [[457, 289], [465, 278], [467, 278],
[470, 277], [461, 279], [461, 279],
[461, 279], [485, 277], [474, 278],
[468, 296], [463, 297], [542, 324],
[449, 327], [614, 321], [376, 318],
[680, 322], [312, 310], [697, 320],
[293, 305], [699, 314], [289, 302],
[693, 316], [296, 305], [515, 451],
[467, 453]]
const EXPECTED_FULLBODY_LANDMARKS = [[460, 287], [469, 277], [472, 276],
[475, 276], [464, 277], [463, 277],
[463, 276], [492, 277], [472, 277],
[471, 295], [465, 295], [542, 323],
[448, 318], [619, 319], [372, 313],
[695, 316], [296, 308], [717, 313],
[273, 304], [718, 304], [280, 298],
[709, 307], [289, 303], [521, 470],
[459, 466], [626, 533], [364, 500],
[704, 616], [347, 614], [710, 631],
[357, 633], [737, 625], [306, 639]];

describeWithFlags('Blazepose', ALL_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let startTensors: number;

  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  beforeEach(async () => {
    startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const modelConfig: poseDetection.BlazeposeModelConfig = {
      quantBytes: 4,
      upperBodyOnly: false
    };
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MediapipeBlazepose, modelConfig);
  });

  it('estimatePoses does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimatePoses(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });
});

describeWithFlags('Blazepose static image ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let image: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    image = await loadImage('pose.jpg', 1000, 667);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  modelType.forEach(type => {
    it('test.', async () => {
      const upperBodyOnly = type === 'upperbody' ? true : false;

      const startTensors = tf.memory().numTensors;

      // Note: this makes a network request for model assets.
      const modelConfig: poseDetection.BlazeposeModelConfig = {
        quantBytes: 4,
        upperBodyOnly
      };
      detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MediapipeBlazepose, modelConfig);

      const beforeTensors = tf.memory().numTensors;

      const result = await detector.estimatePoses(image, {maxPoses: 1, flipHorizontal: false, enableSmoothing: false} as poseDetection.BlazeposeEstimationConfig);
      const xy = result[0].keypoints.map((keypoint) => [keypoint.x, keypoint.y]);
      const expected = upperBodyOnly ? EXPECTED_UPPERBODY_LANDMARKS : EXPECTED_FULLBODY_LANDMARKS;
      expectArraysClose(xy, expected, 10);

      expect(tf.memory().numTensors).toEqual(beforeTensors);

      detector.dispose();

      expect(tf.memory().numTensors).toEqual(startTensors);
    });
  })
});
