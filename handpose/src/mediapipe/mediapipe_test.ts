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

import {loadImage} from '@tensorflow-models/util';
// tslint:disable-next-line: no-imports-from-dist
import {BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as handDetection from '../index';

import {MPHandsMediaPipeModelConfig} from './types';

const MEDIAPIPE_MODEL_CONFIG: MPHandsMediaPipeModelConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/hands'
};

// In pixels.
const EPSILON_IMAGE = 18;
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands_test.py
const EXPECTED_HAND_KEYPOINTS_PREDICTION = [
  [
    [144, 345], [211, 323], [257, 286], [289, 237], [322, 203], [219, 216],
    [238, 138], [249, 90],  [253, 51],  [177, 204], [184, 115], [187, 60],
    [185, 19],  [138, 208], [131, 127], [124, 77],  [117, 36],  [106, 222],
    [92, 159],  [79, 124],  [68, 93]
  ],
  [
    [577, 37],  [504, 56],  [459, 94],  [429, 146], [397, 182], [496, 167],
    [479, 245], [469, 292], [464, 330], [540, 177], [534, 265], [533, 319],
    [536, 360], [581, 172], [587, 252], [593, 304], [599, 346], [615, 157],
    [628, 223], [638, 258], [648, 288]
  ]
];

describeWithFlags('MediaPipe Hands multi hands ', BROWSER_ENVS, () => {
  let detector: handDetection.HandDetector;
  let image: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('hands.jpg', 720, 382);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('test.', async () => {
    // Note: this makes a network request for model assets.
    const model = handDetection.SupportedModels.MPHands;
    detector =
        await handDetection.createDetector(model, MEDIAPIPE_MODEL_CONFIG);

    const result = await detector.estimateHands(image, {staticImageMode: true});
    const keypoints = result.map(
        hand => hand.keypoints.map(keypoint => [keypoint.x, keypoint.y]));

    expectArraysClose(
        keypoints, EXPECTED_HAND_KEYPOINTS_PREDICTION, EPSILON_IMAGE);
    detector.dispose();
  });
});
