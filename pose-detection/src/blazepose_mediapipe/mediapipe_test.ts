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
import {toImageDataLossy} from '../shared/calculators/mask_util';
import {imageToBooleanMask, KARMA_SERVER, loadImage, segmentationIOU} from '../shared/test_util';
import {EXPECTED_LANDMARKS, EXPECTED_WORLD_LANDMARKS, getXYPerFrame, loadVideo} from '../test_util';

import {BlazePoseMediaPipeModelConfig} from './types';

const MEDIAPIPE_MODEL_CONFIG: BlazePoseMediaPipeModelConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/pose',
  enableSegmentation: true,
};

// Measured in pixels.
const EPSILON_IMAGE = 11;
// Measured in meters.
const EPSILON_IMAGE_WORLD = 0.11;
// Measured in pixels.
const EPSILON_VIDEO = 31;
// Measured in meters.
const EPSILON_VIDEO_WORLD = 0.24;
// Measured in percent.
const EPSILON_IOU = 0.88;

describeWithFlags('MediaPipe Pose static image ', BROWSER_ENVS, () => {
  let detector: poseDetection.PoseDetector;
  let image: HTMLImageElement;
  let segmentationImage: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('pose.jpg', 1000, 667);
    segmentationImage = await loadImage('pose_segmentation.png', 1000, 667);
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

    const segmentation = result[0].segmentation;
    const maskValuesToLabel = Array.from(
        Array(256).keys(), (v, _) => segmentation.maskValueToLabel(v));
    const mask = segmentation.mask;
    const actualBooleanMask = imageToBooleanMask(
        (await segmentation.mask.toImageData()).data, 255, 0, 0);
    const expectedBooleanMask = imageToBooleanMask(
        (await toImageDataLossy(segmentationImage)).data, 0, 0, 255);

    expectArraysClose(xy, EXPECTED_LANDMARKS, EPSILON_IMAGE);
    expectArraysClose(xyz, EXPECTED_WORLD_LANDMARKS, EPSILON_IMAGE_WORLD);

    expect(maskValuesToLabel.every(label => label === 'person'));
    expect(mask.getUnderlyingType() === 'canvasimagesource');
    expect(segmentationIOU(expectedBooleanMask, actualBooleanMask))
        .toBeGreaterThanOrEqual(EPSILON_IOU);
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
