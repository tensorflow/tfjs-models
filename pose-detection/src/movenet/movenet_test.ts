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
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {COCO_KEYPOINTS} from '../constants';
import * as poseDetection from '../index';
import {KARMA_SERVER, loadVideo} from '../shared/test_util';

import {SINGLEPOSE_LIGHTNING} from './constants';

interface NamedKeypoint {
  [name: string]: {x: number, y: number};
}

const EPSILON_VIDEO = 60;

describeWithFlags('MoveNet', ALL_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 300000;  // 5mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  beforeEach(async () => {
    // Note: this makes a network request for model assets.
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {modelType: SINGLEPOSE_LIGHTNING});
  });

  it('estimatePoses does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimatePoses(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });
});

describeWithFlags('MoveNet video ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;
  let expected: number[][][];

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    expected = await fetch(`${KARMA_SERVER}/pose_1.json`)
                   .then(response => response.json())
                   .then((result) => {
                     return (result as NamedKeypoint[]).map(namedKeypoint => {
                       return COCO_KEYPOINTS.map(name => {
                         const keypoint = namedKeypoint[name];
                         return [keypoint.x, keypoint.y];
                       });
                     });
                   });
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('test lightning.', async () => {
    // Note: this makes a network request for model assets.

    const model = poseDetection.SupportedModels.MoveNet;
    detector = await poseDetection.createDetector(
        model, {modelType: SINGLEPOSE_LIGHTNING});

    const result: number[][][] = [];

    const callback = async(video: HTMLVideoElement, timestamp: number):
        Promise<poseDetection.Keypoint[]> => {
          const poses =
              await detector.estimatePoses(video, null /* config */, timestamp);
          result.push(poses[0].keypoints.map(kp => [kp.x, kp.y]));
          return poses[0].keypoints;
        };

    // We set the timestamp increment to 33333 microseconds to simulate
    // the 30 fps video input. We do this so that the filter uses the
    // same fps as the reference test.
    // https://github.com/google/mediapipe/blob/ecb5b5f44ab23ea620ef97a479407c699e424aa7/mediapipe/python/solution_base.py#L297
    const simulatedInterval = 33.3333;

    // Synthetic video at 30FPS.
    await loadVideo(
        'pose_1.mp4', 30 /* fps */, callback, expected,
        poseDetection.util.getAdjacentPairs(model), simulatedInterval);

    expectArraysClose(result, expected, EPSILON_VIDEO);

    detector.dispose();
  });
});
