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
import {KARMA_SERVER, loadImage} from '../shared/test_util';
import {getXYPerFrame, loadVideo} from '../test_util';

import {BlazePoseMediaPipeModelConfig} from './types';

const MEDIAPIPE_MODEL_CONFIG: BlazePoseMediaPipeModelConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/pose'
};

// Measured in pixels.
const EPSILON_IMAGE = 11;
// Measured in meters.
const EPSILON_IMAGE_WORLD = 0.1;
// Measured in pixels.
const EPSILON_VIDEO = 31;
// Measured in meters.
const EPSILON_VIDEO_WORLD = 0.24;
// ref:
// https://github.com/google/mediapipe/blob/7c331ad58b2cca0dca468e342768900041d65adc/mediapipe/python/solutions/pose_test.py#L31-L51
const EXPECTED_LANDMARKS = [
  [463, 284], [470, 275], [473, 274], [476, 273], [464, 275], [464, 275],
  [463, 275], [494, 275], [473, 276], [470, 293], [464, 293], [544, 322],
  [450, 325], [626, 321], [371, 316], [697, 314], [300, 305], [717, 313],
  [279, 305], [719, 311], [279, 302], [711, 313], [285, 304], [521, 471],
  [465, 466], [608, 548], [357, 492], [701, 614], [350, 608], [707, 624],
  [362, 628], [732, 632], [306, 630]
];
const EXPECTED_WORLD_LANDMARKS = [
  [-0.12, -0.58, -0.19], [-0.1, -0.62, -0.19],  [-0.1, -0.62, -0.19],
  [-0.1, -0.63, -0.19],  [-0.12, -0.62, -0.17], [-0.12, -0.62, -0.17],
  [-0.12, -0.62, -0.17], [0., -0.63, -0.16],    [-0.08, -0.63, -0.08],
  [-0.07, -0.57, -0.17], [-0.1, -0.56, -0.15],  [0.16, -0.49, -0.1],
  [-0.15, -0.49, -0.03], [0.41, -0.49, -0.15],  [-0.42, -0.52, -0.04],
  [0.65, -0.48, -0.2],   [-0.64, -0.52, -0.14], [0.71, -0.5, -0.21],
  [-0.71, -0.54, -0.16], [0.72, -0.51, -0.24],  [-0.7, -0.55, -0.19],
  [0.66, -0.49, -0.21],  [-0.65, -0.53, -0.16], [0.1, 0.01, -0.04],
  [-0.1, -0.01, 0.04],   [0.36, 0.21, -0.1],    [-0.45, 0.05, -0.1],
  [0.68, 0.44, 0.01],    [-0.49, 0.43, -0.],    [0.72, 0.46, 0.01],
  [-0.5, 0.47, 0.01],    [0.78, 0.46, -0.1],    [-0.61, 0.51, -0.05],
];

describeWithFlags('MediaPipe Pose static image ', BROWSER_ENVS, () => {
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

  it('test.', async () => {
    // Note: this makes a network request for model assets.
    const model = poseDetection.SupportedModels.BlazePose;
    detector =
        await poseDetection.createDetector(model, MEDIAPIPE_MODEL_CONFIG);

    const result = await detector.estimatePoses(image, {});
    const xy = result[0].keypoints.map((keypoint) => [keypoint.x, keypoint.y]);
    const xyz = result[0].keypoints3D.map(
        (keypoint) => [keypoint.x, keypoint.y, keypoint.z]);

    expectArraysClose(xy, EXPECTED_LANDMARKS, EPSILON_IMAGE);
    expectArraysClose(xyz, EXPECTED_WORLD_LANDMARKS, EPSILON_IMAGE_WORLD);
    detector.dispose();
  });
});

describeWithFlags('MediaPipe Pose video ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let timeout: number;
  let expected: number[][][];
  let expected3D: number[][][];

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    expected = await fetch(`${KARMA_SERVER}/pose_squats.full.json`)
                   .then(response => response.json())
                   .then(result => getXYPerFrame(result));

    expected3D = await fetch(`${KARMA_SERVER}/pose_squats_3d.full.json`)
                     .then(response => response.json());
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
    const result3D: number[][][] = [];

    const callback = async(video: HTMLVideoElement, timestamp: number):
        Promise<poseDetection.Pose[]> => {
          const poses =
              await detector.estimatePoses(video, null /* config */, timestamp);
          // BlazePose only returns single pose for now.
          result.push(poses[0].keypoints.map(kp => [kp.x, kp.y]));
          result3D.push(poses[0].keypoints3D.map(kp => [kp.x, kp.y, kp.z]));

          return poses;
        };

    // Original video source in 720 * 1280 resolution:
    // https://www.pexels.com/video/woman-doing-squats-4838220/ Video is
    // compressed to be smaller with less frames (5fps), using below command:
    // `ffmpeg -i original_pose.mp4 -r 5 -vcodec libx264 -crf 28 -profile:v
    // baseline pose_squats.mp4`
    await loadVideo('pose_squats.mp4', 5 /* fps */, callback, expected, model);

    expectArraysClose(result, expected, EPSILON_VIDEO);
    expectArraysClose(result3D, expected3D, EPSILON_VIDEO_WORLD);

    detector.dispose();
  });
});
