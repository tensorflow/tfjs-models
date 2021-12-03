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
import {MEDIAPIPE_CONNECTED_KEYPOINTS_PAIRS} from '../constants';

import * as handPoseDetection from '../index';
import {getXYPerFrame, KARMA_SERVER, loadImage, loadVideo} from '../shared/test_util';

// Measured in pixels.
const EPSILON_IMAGE = 12;
// Measured in pixels.
const EPSILON_VIDEO = 18;
// Measured in meters.
const EPSILON_VIDEO_WORLD = 0.01;

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands_test.py
const EXPECTED_HAND_KEYPOINTS_PREDICTION = [
  [
    [580, 34],  [504, 50],  [459, 94],  [429, 146], [397, 182], [507, 167],
    [479, 245], [469, 292], [464, 330], [545, 180], [534, 265], [533, 319],
    [536, 360], [581, 172], [587, 252], [593, 304], [599, 346], [615, 168],
    [628, 223], [638, 258], [648, 288]
  ],
  [
    [138, 343], [211, 330], [257, 286], [289, 237], [322, 203], [219, 216],
    [238, 138], [249, 90],  [253, 51],  [177, 204], [184, 115], [187, 60],
    [185, 19],  [138, 208], [131, 127], [124, 77],  [117, 36],  [106, 222],
    [92, 159],  [79, 124],  [68, 93]
  ]
];

describeWithFlags('MediaPipeHands', ALL_ENVS, () => {
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('estimateHands does not leak memory.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const detector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands, {runtime: 'tfjs'});
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimateHands(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });

  it('throws error when runtime is not set.', async (done) => {
    try {
      await handPoseDetection.createDetector(
          handPoseDetection.SupportedModels.MediaPipeHands);
      done.fail('Loading without runtime succeeded unexpectedly.');
    } catch (e) {
      expect(e.message).toEqual(
          `Expect modelConfig.runtime to be either ` +
          `'tfjs' or 'mediapipe', but got undefined`);
      done();
    }
  });
});

describeWithFlags('MediaPipeHands static image ', BROWSER_ENVS, () => {
  let detector: handPoseDetection.HandDetector;
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

  it('test lite model.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    detector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {runtime: 'tfjs', modelType: 'lite'});

    const beforeTensors = tf.memory().numTensors;

    const result = await detector.estimateHands(image, {
      staticImageMode: true
    } as handPoseDetection.MediaPipeHandsTfjsEstimationConfig);
    const keypoints = result.map(
        hand => hand.keypoints.map(keypoint => [keypoint.x, keypoint.y]));

    expectArraysClose(
        keypoints, EXPECTED_HAND_KEYPOINTS_PREDICTION, EPSILON_IMAGE);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });

  it('test full model.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    detector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {runtime: 'tfjs', modelType: 'full'});

    const beforeTensors = tf.memory().numTensors;

    const result = await detector.estimateHands(image, {
      staticImageMode: true
    } as handPoseDetection.MediaPipeHandsTfjsEstimationConfig);
    const keypoints = result.map(
        hand => hand.keypoints.map(keypoint => [keypoint.x, keypoint.y]));

    expectArraysClose(
        keypoints, EXPECTED_HAND_KEYPOINTS_PREDICTION, EPSILON_IMAGE);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });
});

describeWithFlags('MediaPipe Hands video ', BROWSER_ENVS, () => {
  let detector: handPoseDetection.HandDetector;
  let timeout: number;
  let expected: number[][][];
  let expected3D: number[][][];

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    expected = await fetch(`${KARMA_SERVER}/asl_hand.full.json`)
                   .then(response => response.json())
                   .then(result => getXYPerFrame(result));

    expected3D = await fetch(`${KARMA_SERVER}/asl_hand_3d.full.json`)
                     .then(response => response.json());
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('test.', async () => {
    // Note: this makes a network request for model assets.
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    detector = await handPoseDetection.createDetector(
        model, {runtime: 'tfjs', maxHands: 1});

    const result: number[][][] = [];
    const result3D: number[][][] = [];

    const callback = async(video: HTMLVideoElement, timestamp: number):
        Promise<handPoseDetection.Keypoint[]> => {
          const hands = await detector.estimateHands(video, null /* config */);

          // maxNumHands is set to 1.
          result.push(hands[0].keypoints.map(kp => [kp.x, kp.y]));
          result3D.push(hands[0].keypoints3D.map(kp => [kp.x, kp.y, kp.z]));

          return hands[0].keypoints;
        };

    await loadVideo(
        'asl_hand.25fps.mp4', 25 /* fps */, callback, expected,
        MEDIAPIPE_CONNECTED_KEYPOINTS_PAIRS, 0 /* simulatedInterval unused */);

    expectArraysClose(result, expected, EPSILON_VIDEO);
    expectArraysClose(result3D, expected3D, EPSILON_VIDEO_WORLD);

    detector.dispose();
  });
});
