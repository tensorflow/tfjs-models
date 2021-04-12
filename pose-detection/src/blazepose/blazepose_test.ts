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

import * as poseDetection from '../index';
import {KARMA_SERVER, loadImage, loadVideo} from '../test_util';

const UPPERBODY_ONLY = [false];
const EPSILON = 10;
// ref:
// https://github.com/google/mediapipe/blob/7c331ad58b2cca0dca468e342768900041d65adc/mediapipe/python/solutions/pose_test.py#L31-L51
const EXPECTED_UPPERBODY_LANDMARKS = [
  [457, 289], [465, 278], [467, 278], [470, 277], [461, 279],
  [461, 279], [461, 279], [485, 277], [474, 278], [468, 296],
  [463, 297], [542, 324], [449, 327], [614, 321], [376, 318],
  [680, 322], [312, 310], [697, 320], [293, 305], [699, 314],
  [289, 302], [693, 316], [296, 305], [515, 451], [467, 453]
];
const EXPECTED_FULLBODY_LANDMARKS = [
  [460, 287], [469, 277], [472, 276], [475, 276], [464, 277], [463, 277],
  [463, 276], [492, 277], [472, 277], [471, 295], [465, 295], [542, 323],
  [448, 318], [619, 319], [372, 313], [695, 316], [296, 308], [717, 313],
  [273, 304], [718, 304], [280, 298], [709, 307], [289, 303], [521, 470],
  [459, 466], [626, 533], [364, 500], [704, 616], [347, 614], [710, 631],
  [357, 633], [737, 625], [306, 639]
];

describeWithFlags('Blazepose', ALL_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let startTensors: number;

  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 300000;  // 5mins
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

  UPPERBODY_ONLY.forEach(upperBodyOnly => {
    it('test.', async () => {
      const startTensors = tf.memory().numTensors;

      // Note: this makes a network request for model assets.
      const modelConfig:
          poseDetection.BlazeposeModelConfig = {quantBytes: 4, upperBodyOnly};
      detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MediapipeBlazepose, modelConfig);

      const beforeTensors = tf.memory().numTensors;

      const result = await detector.estimatePoses(
          image,
          {maxPoses: 1, flipHorizontal: false, enableSmoothing: false} as
              poseDetection.BlazeposeEstimationConfig);
      const xy =
          result[0].keypoints.map((keypoint) => [keypoint.x, keypoint.y]);
      const expected = upperBodyOnly ? EXPECTED_UPPERBODY_LANDMARKS :
                                       EXPECTED_FULLBODY_LANDMARKS;
      expectArraysClose(xy, expected, EPSILON);

      expect(tf.memory().numTensors).toEqual(beforeTensors);

      detector.dispose();

      expect(tf.memory().numTensors).toEqual(startTensors);
    });
  });
});

describeWithFlags('Blazepose video ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;
  let expectedFull: number[][][];

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    expectedFull = await fetch(`${KARMA_SERVER}/pose_squats.full_body.json`)
                       .then(response => response.json());
    expectedFull = expectedFull.map(frameResult => {
      return frameResult.map(keypoint => [keypoint[0], keypoint[1]]);
    });
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  UPPERBODY_ONLY.forEach(upperBodyOnly => {
    it('test.', async () => {
      // Note: this makes a network request for model assets.
      const modelConfig:
          poseDetection.BlazeposeModelConfig = {quantBytes: 4, upperBodyOnly};
      detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MediapipeBlazepose, modelConfig);

      const result: number[][][] = [];

      const callback =
          async(video: HTMLVideoElement, timestamp: number): Promise<void> => {
        const poses =
            await detector.estimatePoses(video, null /* config */, timestamp);
        result.push(poses[0].keypoints.map(kp => [kp.x, kp.y]));
      };

      // Original video source in 720 * 1280 resolution:
      // https://www.pexels.com/video/woman-doing-squats-4838220/ Video is
      // compressed to be smaller with less frames (5fps), using below command:
      // `ffmpeg -i original_pose.mp4 -r 5 -vcodec libx264 -crf 28 -profile:v
      // baseline pose_squats.mp4`
      await loadVideo('pose_squats.mp4', 5 /* fps */, callback);

      let count = 0;
      for (let i = 0; i < result.length; i++) {
        const frameResult = result[i];
        for (let j = 0; j < frameResult.length; j++) {
          const keypoint = frameResult[j];
          const x = keypoint[0];
          const y = keypoint[1];

          const ex = expectedFull[i][j][0];
          const ey = expectedFull[i][j][1];

          if (Math.abs(x - ex) > 30 || Math.abs(y - ey) > 30) {
            count++;
            console.log(`Frame ${i}, Keypoint ${j}: Actual [${x}, ${
                y}], Expected [${ex}, ${ey}]`);
          }
        }
      }
      console.log(count);
      // expectArraysClose(result.slice(0, 65), expectedFull, 45);

      detector.dispose();
    });
  });
});
