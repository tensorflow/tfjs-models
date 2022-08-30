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

import {EXPECTED_LANDMARKS, EXPECTED_WORLD_LANDMARKS} from '../blazepose_mediapipe/mediapipe_test';
import * as poseDetection from '../index';
import {toImageDataLossy} from '../shared/calculators/mask_util';
import {getXYPerFrame, imageToBooleanMask, KARMA_SERVER, loadImage, loadVideo, segmentationIOU} from '../shared/test_util';

// Measured in pixels.
const EPSILON_IMAGE = 15;
// Measured in meters.
const EPSILON_IMAGE_WORLD = 0.21;
// Measured in pixels.
const EPSILON_VIDEO = 28;
// Measured in meters.
const EPSILON_VIDEO_WORLD = 0.19;
// Measured in percent.
const EPSILON_IOU = 0.94;

describeWithFlags('BlazePose', ALL_ENVS, () => {
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('estimatePoses does not leak memory with segmentation off.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        {runtime: 'tfjs', enableSegmentation: false});
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimatePoses(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });

  it('throws error when runtime is not set.', async (done) => {
    try {
      await poseDetection.createDetector(
          poseDetection.SupportedModels.BlazePose);
      done.fail('Loading without runtime succeeded unexpectedly.');
    } catch (e) {
      expect(e.message).toEqual(
          `Expect modelConfig.runtime to be either ` +
          `'tfjs' or 'mediapipe', but got undefined`);
      done();
    }
  });
});

async function expectModel(
    image: HTMLImageElement, segmentationImage: HTMLImageElement,
    modelType: poseDetection.BlazePoseModelType) {
  const startTensors = tf.memory().numTensors;

  // Note: this makes a network request for model assets.
  const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.BlazePose, {
        runtime: 'tfjs',
        modelType,
        enableSmoothing: true,
        enableSegmentation: true,
        smoothSegmentation: false
      });

  const beforeTensors = tf.memory().numTensors;

  const result = await detector.estimatePoses(
      image,
      {runtime: 'tfjs', maxPoses: 1, flipHorizontal: false} as
          poseDetection.BlazePoseTfjsEstimationConfig);
  const xy = result[0].keypoints.map((keypoint) => [keypoint.x, keypoint.y]);
  const worldXyz = result[0].keypoints3D.map(
      (keypoint) => [keypoint.x, keypoint.y, keypoint.z]);

  const segmentation = result[0].segmentation;
  const maskValuesToLabel =
      Array.from(Array(256).keys(), (v, _) => segmentation.maskValueToLabel(v));
  const mask = segmentation.mask;
  const actualBooleanMask = imageToBooleanMask(
      // Round to binary mask using red value cutoff of 128.
      (await segmentation.mask.toImageData()).data, 128, 0, 0);
  const expectedBooleanMask = imageToBooleanMask(
      (await toImageDataLossy(segmentationImage)).data, 0, 0, 255);

  expectArraysClose(xy, EXPECTED_LANDMARKS, EPSILON_IMAGE);
  expectArraysClose(worldXyz, EXPECTED_WORLD_LANDMARKS, EPSILON_IMAGE_WORLD);

  expect(maskValuesToLabel.every(label => label === 'person'));
  expect(mask.getUnderlyingType() === 'tensor');
  expect(segmentationIOU(expectedBooleanMask, actualBooleanMask))
      .toBeGreaterThanOrEqual(EPSILON_IOU);

  tf.dispose([await segmentation.mask.toTensor()]);
  expect(tf.memory().numTensors).toEqual(beforeTensors);

  detector.dispose();

  expect(tf.memory().numTensors).toEqual(startTensors);
}

describeWithFlags('BlazePose static image ', BROWSER_ENVS, () => {
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

  it('estimatePoses does not leak memory with segmentation on.', async () => {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        {runtime: 'tfjs', enableSegmentation: true, smoothSegmentation: true});

    const beforeTensors = tf.memory().numTensors;

    let output = await detector.estimatePoses(image);
    (await output[0].segmentation.mask.toTensor()).dispose();

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    // Call again to test smoothing code.
    output = await detector.estimatePoses(image);
    (await output[0].segmentation.mask.toTensor()).dispose();

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    (await output[0].segmentation.mask.toTensor()).dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  });

  it('test lite model.', async () => {
    await expectModel(image, segmentationImage, 'lite');
  });

  it('test full model.', async () => {
    await expectModel(image, segmentationImage, 'full');
  });

  it('test heavy model.', async () => {
    await expectModel(image, segmentationImage, 'heavy');
  });
});

describeWithFlags('BlazePose video ', BROWSER_ENVS, () => {
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
    detector = await poseDetection.createDetector(model, {runtime: 'tfjs'});

    const result: number[][][] = [];
    const result3D: number[][][] = [];

    const callback = async(video: HTMLVideoElement, timestamp: number):
        Promise<poseDetection.Keypoint[]> => {
          const poses =
              await detector.estimatePoses(video, null /* config */, timestamp);
          // BlazePose only returns single pose for now.
          result.push(poses[0].keypoints.map(kp => [kp.x, kp.y]));
          result3D.push(poses[0].keypoints3D.map(kp => [kp.x, kp.y, kp.z]));

          return poses[0].keypoints;
        };

    // We set the timestamp increment to 33333 microseconds to simulate
    // the 30 fps video input. We do this so that the filter uses the
    // same fps as the reference test.
    // https://github.com/google/mediapipe/blob/ecb5b5f44ab23ea620ef97a479407c699e424aa7/mediapipe/python/solution_base.py#L297
    const simulatedInterval = 33.3333;

    // Original video source in 720 * 1280 resolution:
    // https://www.pexels.com/video/woman-doing-squats-4838220/ Video is
    // compressed to be smaller with less frames (5fps), using below
    // command:
    // `ffmpeg -i original_pose.mp4 -r 5 -vcodec libx264 -crf 28 -profile:v
    // baseline pose_squats.mp4`
    await loadVideo(
        'pose_squats.mp4', 5 /* fps */, callback, expected,
        poseDetection.util.getAdjacentPairs(model), simulatedInterval);

    expectArraysClose(result, expected, EPSILON_VIDEO);
    expectArraysClose(result3D, expected3D, EPSILON_VIDEO_WORLD);

    detector.dispose();
  });
});
