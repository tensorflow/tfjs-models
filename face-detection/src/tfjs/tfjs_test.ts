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
import {expectArraysClose, expectArraysEqual, expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as faceDetection from '../index';
import {expectFaceDetector, MEDIAPIPE_MODEL_CONFIG} from '../mediapipe/mediapipe_test';
import {MediaPipeFaceDetectorModelType} from '../mediapipe/types';
import {loadImage} from '../shared/test_util';

const TFJS_MODEL_CONFIG = {
  runtime: 'tfjs' as const ,
  maxFaces: 1
};

// Measured in pixels.
const EPSILON_IMAGE = 10;

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
    const detector = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        {...TFJS_MODEL_CONFIG, modelType});
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);

    const beforeTensors = tf.memory().numTensors;

    await detector.estimateFaces(input);

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

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins
    image = await loadImage('portrait.jpg', 820, 1024);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function expectTFJSFaceDetector(
      modelType: MediaPipeFaceDetectorModelType) {
    // Note: this makes a network request for model assets.
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
    const detector = await faceDetection.createDetector(
        model, {...TFJS_MODEL_CONFIG, modelType});

    return expectFaceDetector(detector, image, modelType);
  }

  it('short range.', async () => {
    await expectTFJSFaceDetector('short');
  });

  it('full range.', async () => {
    await expectTFJSFaceDetector('full');
  });

  it('TFJS and Mediapipe backends match.', async () => {
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
    const tfjsResults =
        await faceDetection.createDetector(model, {...TFJS_MODEL_CONFIG})
            .then(detector => detector.estimateFaces(image));

    const mediapipeResults =
        await faceDetection.createDetector(model, {...MEDIAPIPE_MODEL_CONFIG})
            .then(async detector => {
              await detector.estimateFaces(image);
              // Initialize model.
              return detector.estimateFaces(image);
            });

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

    for (let i = 0; i < tfjsResults.length; i++) {
      for (const key of ['height', 'width', 'xMax', 'xMin', 'yMax', 'yMin'] as
           const ) {
        expectNumbersClose(
            tfjsResults[i].box[key], mediapipeResults[i].box[key],
            EPSILON_IMAGE);
      }
    }
  });
});
