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
import {expectArraysClose, expectArraysEqual} from '@tensorflow/tfjs-core/dist/test_util';

import * as faceDetection from '../index';
import {expectFaceMesh, MEDIAPIPE_MODEL_CONFIG} from '../mediapipe/mediapipe_test';
import {loadImage} from '../shared/test_util';
import {loadDetectorModel} from './detector';
import {MediaPipeFaceDetectorModelType} from './types';

const TFJS_MODEL_CONFIG = {
  runtime: 'tfjs' as const ,
};

const SHORT_RANGE_EXPECTED_FACE_KEY_POINTS =
    [[363, 182], [460, 186], [420, 241], [417, 284], [295, 199], [502, 198]];
const FULL_RANGE_EXPECTED_FACE_KEY_POINTS =
    [[363, 181], [455, 181], [413, 233], [411, 278], [306, 204], [499, 207]];
// Measured in pixels.
const EPSILON_IMAGE = 5;

describeWithFlags('TFJS FaceDetector ', ALL_ENVS, () => {
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function expectTFJSFaceDetector(
      modelType: MediaPipeFaceDetectorModelType) {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const detector = await loadDetectorModel({modelType, maxFaces: 1});
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.detectFaces(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  }

  it('short range detectFaces does not leak memory.', async () => {
    await expectTFJSFaceDetector('short');
  });

  it('full range detectFaces does not leak memory.', async () => {
    await expectTFJSFaceDetector('full');
  });
});

describeWithFlags('TFJS FaceDetector static image ', BROWSER_ENVS, () => {
  let image: HTMLImageElement;
  let timeout: number;

  async function expectTFJSFaceDetector(
      image: HTMLImageElement, modelType: MediaPipeFaceDetectorModelType) {
    // Note: this makes a network request for model assets.
    const detector = await loadDetectorModel({modelType, maxFaces: 1});

    for (let i = 0; i < 5; ++i) {
      const result = await detector.detectFaces(image);
      expect(result.length).toBe(1);

      const keypoints = result[0].locationData.relativeKeypoints.map(
          keypoint => [keypoint.x * image.width, keypoint.y * image.height]);
      expect(keypoints.length).toBe(6);

      const expectedKeypoints = modelType === 'short' ?
          SHORT_RANGE_EXPECTED_FACE_KEY_POINTS :
          FULL_RANGE_EXPECTED_FACE_KEY_POINTS;

      expectArraysClose(keypoints, expectedKeypoints, EPSILON_IMAGE);
    }
  }

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('portrait.jpg', 820, 1024);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('short range.', async () => {
    await expectTFJSFaceDetector(image, 'short');
  });

  it('full range.', async () => {
    await expectTFJSFaceDetector(image, 'full');
  });
});

describeWithFlags('TFJS FaceMesh ', ALL_ENVS, () => {
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function expectTFJSFaceMesh(refineLandmarks: boolean) {
    const startTensors = tf.memory().numTensors;

    // Note: this makes a network request for model assets.
    const detector = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceMesh,
        {...TFJS_MODEL_CONFIG, refineLandmarks});
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimateFaces(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);

    detector.dispose();
    input.dispose();

    expect(tf.memory().numTensors).toEqual(startTensors);
  }

  it('with attention estimateFaces does not leak memory.', async () => {
    await expectTFJSFaceMesh(false);
  });

  it('without attention estimateFaces does not leak memory.', async () => {
    await expectTFJSFaceMesh(true);
  });

  it('throws error when runtime is not set.', async (done) => {
    try {
      await faceDetection.createDetector(
          faceDetection.SupportedModels.MediaPipeFaceMesh);
      done.fail('Loading without runtime succeeded unexpectedly.');
    } catch (e) {
      expect(e.message).toEqual(
          `Expect modelConfig.runtime to be either ` +
          `'tfjs' or 'mediapipe', but got undefined`);
      done();
    }
  });
});

describeWithFlags('TFJS FaceMesh static image ', BROWSER_ENVS, () => {
  let image: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('portrait.jpg', 820, 1024);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function expectTFJSFaceMesh(
      image: HTMLImageElement, staticImageMode: boolean,
      refineLandmarks: boolean, numFrames: number) {
    // Note: this makes a network request for model assets.
    const model = faceDetection.SupportedModels.MediaPipeFaceMesh;
    const detector = await faceDetection.createDetector(
        model, {...TFJS_MODEL_CONFIG, refineLandmarks});

    await expectFaceMesh(
        detector, image, staticImageMode, refineLandmarks, numFrames,
        EPSILON_IMAGE);
  }

  it('static image mode no attention.', async () => {
    await expectTFJSFaceMesh(image, true, false, 5);
  });

  it('static image mode with attention.', async () => {
    await expectTFJSFaceMesh(image, true, true, 5);
  });

  it('streaming mode no attention.', async () => {
    await expectTFJSFaceMesh(image, false, false, 10);
  });

  it('streaming mode with attention.', async () => {
    await expectTFJSFaceMesh(image, false, true, 10);
  });

  it('TFJS and Mediapipe backends match.', async () => {
    const model = faceDetection.SupportedModels.MediaPipeFaceMesh;
    const tfjsResults =
        await faceDetection
            .createDetector(
                model, {...TFJS_MODEL_CONFIG, refineLandmarks: true})
            .then(detector => detector.estimateFaces(image));

    const mediapipeResults =
        await faceDetection
            .createDetector(
                model, {...MEDIAPIPE_MODEL_CONFIG, refineLandmarks: true})
            .then(detector => detector.estimateFaces(image));

    expect(tfjsResults.length).toBe(mediapipeResults.length);

    const tfjsKeypoints =
        tfjsResults
            .map(
                face => face.keypoints.map(
                    keypoint => [keypoint.x, keypoint.y,
                                 keypoint.name] as [number, number, string]))
            .flat();

    const mediapipeKeypoints =
        mediapipeResults
            .map(
                face => face.keypoints.map(
                    keypoint => [keypoint.x, keypoint.y,
                                 keypoint.name] as [number, number, string]))
            .flat();

    expectArraysClose(
        tfjsKeypoints.map(keypoint => [keypoint[0], keypoint[1]]),
        mediapipeKeypoints.map(keypoint => [keypoint[0], keypoint[1]]),
        EPSILON_IMAGE);
    expectArraysEqual(
        tfjsKeypoints.map(keypoint => keypoint[2]),
        mediapipeKeypoints.map(keypoint => keypoint[2]));
  });
});
