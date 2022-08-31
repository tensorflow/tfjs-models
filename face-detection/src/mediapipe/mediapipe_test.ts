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
import {expectArraysClose, expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as faceDetection from '../index';
import {BoundingBox} from '../shared/calculators/interfaces/shape_interfaces';
import {loadImage} from '../shared/test_util';
import {MediaPipeFaceDetectorModelType} from './types';

export const MEDIAPIPE_MODEL_CONFIG = {
  runtime: 'mediapipe' as const ,
  solutionPath: 'base/node_modules/@mediapipe/face_detection',
  maxFaces: 1
};

const SHORT_RANGE_EXPECTED_FACE_KEY_POINTS =
    [[363, 182], [460, 186], [420, 241], [417, 284], [295, 199], [502, 198]];
const FULL_RANGE_EXPECTED_FACE_KEY_POINTS =
    [[363, 181], [455, 181], [413, 233], [411, 278], [306, 204], [499, 207]];
const SHORT_RANGE_EXPECTED_BOX: BoundingBox = {
  xMin: 282,
  xMax: 520,
  yMin: 113,
  yMax: 351,
  width: 238,
  height: 238
};
const FULL_RANGE_EXPECTED_BOX: BoundingBox = {
  xMin: 292,
  xMax: 526,
  yMin: 106,
  yMax: 339,
  width: 234,
  height: 233
};
// Measured in pixels.
const EPSILON_IMAGE = 10;

export async function expectFaceDetector(
    detector: faceDetection.FaceDetector, image: HTMLImageElement,
    modelType: MediaPipeFaceDetectorModelType) {
  // Initialize model.
  await detector.estimateFaces(image);
  for (let i = 0; i < 5; ++i) {
    const result = await detector.estimateFaces(image);
    expect(result.length).toBe(1);

    const box = result[0].box;
    const expectedBox = modelType === 'short' ? SHORT_RANGE_EXPECTED_BOX :
                                                FULL_RANGE_EXPECTED_BOX;

    expectNumbersClose(box.xMin, expectedBox.xMin, EPSILON_IMAGE);
    expectNumbersClose(box.xMax, expectedBox.xMax, EPSILON_IMAGE);
    expectNumbersClose(box.yMin, expectedBox.yMin, EPSILON_IMAGE);
    expectNumbersClose(box.yMax, expectedBox.yMax, EPSILON_IMAGE);
    expectNumbersClose(box.width, expectedBox.width, EPSILON_IMAGE);
    expectNumbersClose(box.height, expectedBox.height, EPSILON_IMAGE);

    const keypoints =
        result[0].keypoints.map(keypoint => [keypoint.x, keypoint.y]);
    expect(keypoints.length).toBe(6);

    const expectedKeypoints = modelType === 'short' ?
        SHORT_RANGE_EXPECTED_FACE_KEY_POINTS :
        FULL_RANGE_EXPECTED_FACE_KEY_POINTS;

    expectArraysClose(keypoints, expectedKeypoints, EPSILON_IMAGE);
  }
}
describeWithFlags('MediaPipe FaceDetector static image ', BROWSER_ENVS, () => {
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

  async function expectMediaPipeFaceDetector(
      modelType: MediaPipeFaceDetectorModelType) {
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
    const detector = await faceDetection.createDetector(
        model, {...MEDIAPIPE_MODEL_CONFIG, modelType});

    await expectFaceDetector(detector, image, modelType);

    detector.dispose();
  }

  it('short range.', async () => {
    await expectMediaPipeFaceDetector('short');
  });

  it('full range.', async () => {
    await expectMediaPipeFaceDetector('full');
  });
});
