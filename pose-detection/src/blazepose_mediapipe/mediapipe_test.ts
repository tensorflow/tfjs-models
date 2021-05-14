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

// tslint:disable-next-line: no-imports-from-dist
import {BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as poseDetection from '../index';
import {getXYPerFrame, KARMA_SERVER, loadVideo} from '../test_util';
import {BlazePoseMediaPipeModelConfig} from './types';

const MEDIAPIPE_MODEL_CONFIG: BlazePoseMediaPipeModelConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/pose'
};

// const EPSILON_IMAGE = 30;
const EPSILON_VIDEO = 75;
// ref:
// https://github.com/google/mediapipe/blob/7c331ad58b2cca0dca468e342768900041d65adc/mediapipe/python/solutions/pose_test.py#L31-L51
// const EXPECTED_LANDMARKS = [
//   [460, 287], [469, 277], [472, 276], [475, 276], [464, 277], [463, 277],
//   [463, 276], [492, 277], [472, 277], [471, 295], [465, 295], [542, 323],
//   [448, 318], [619, 319], [372, 313], [695, 316], [296, 308], [717, 313],
//   [273, 304], [718, 304], [280, 298], [709, 307], [289, 303], [521, 470],
//   [459, 466], [626, 533], [364, 500], [704, 616], [347, 614], [710, 631],
//   [357, 633], [737, 625], [306, 639]
// ];

// Disable this test until @mediapipe/pose bug is fixed.
// describeWithFlags('MediaPipe Pose static image ', BROWSER_ENVS, () => {
//   let detector: poseDetection.PoseDetector;
//   let image: HTMLImageElement;
//   let timeout: number;

//   beforeAll(async () => {
//     timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
//     jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
//   });

//   afterAll(() => {
//     jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
//   });

//   it('test.', async () => {
//     // Note: this makes a network request for model assets.
//     const model = poseDetection.SupportedModels.BlazePose;
//     detector =
//         await poseDetection.createDetector(model, MEDIAPIPE_MODEL_CONFIG);
//     image = await loadImage('pose.jpg', 1000, 667);

//     const result = await detector.estimatePoses(image, {});
//     const xy = result[0].keypoints.map((keypoint) => [keypoint.x,
//     keypoint.y]); const expected = EXPECTED_LANDMARKS; expectArraysClose(xy,
//     expected, EPSILON_IMAGE); detector.dispose();
//   });
// });

describeWithFlags('MediaPipe Pose video ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;
  let expected: number[][][];

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    expected = await fetch(`${KARMA_SERVER}/pose_squats.full.json`)
                   .then(response => response.json())
                   .then(result => getXYPerFrame(result));
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('test.', async () => {
    // Note: this makes a network request for model assets.
    const model = poseDetection.SupportedModels.BlazePose;
    detector =
        await poseDetection.createDetector(model, MEDIAPIPE_MODEL_CONFIG);

    const result: number[][][] = [];

    const callback = async(video: HTMLVideoElement, timestamp: number):
        Promise<poseDetection.Pose[]> => {
          const poses =
              await detector.estimatePoses(video, null /* config */, timestamp);
          result.push(poses[0].keypoints.map(kp => [kp.x, kp.y]));
          return poses;
        };

    // Original video source in 720 * 1280 resolution:
    // https://www.pexels.com/video/woman-doing-squats-4838220/ Video is
    // compressed to be smaller with less frames (5fps), using below command:
    // `ffmpeg -i original_pose.mp4 -r 5 -vcodec libx264 -crf 28 -profile:v
    // baseline pose_squats.mp4`
    await loadVideo('pose_squats.mp4', 5 /* fps */, callback, expected, model);

    expectArraysClose(result, expected, EPSILON_VIDEO);

    detector.dispose();
  });
});
