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
import {MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS, MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS_WITH_IRISES} from '../constants';

import * as faceLandmarksDetection from '../index';
import {BoundingBox} from '../shared/calculators/interfaces/shape_interfaces';
import {loadImage} from '../shared/test_util';

export const MEDIAPIPE_MODEL_CONFIG = {
  runtime: 'mediapipe' as const ,
  solutionPath: 'base/node_modules/@mediapipe/face_mesh'
};

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_test.py
// Measured in pixels.
const EPSILON_IMAGE = 5;

const EYE_INDICES_TO_LANDMARKS: Array<[number, [number, number]]> = [
  [33, [345, 178]],   //
  [7, [348, 179]],    //
  [163, [352, 178]],  //
  [144, [357, 179]],  //
  [145, [365, 179]],  //
  [153, [371, 179]],  //
  [154, [378, 178]],  //
  [155, [381, 177]],  //
  [133, [383, 177]],  //
  [246, [347, 175]],  //
  [161, [350, 174]],  //
  [160, [355, 172]],  //
  [159, [362, 170]],  //
  [158, [368, 171]],  //
  [157, [375, 172]],  //
  [173, [380, 175]],  //
  [263, [467, 176]],  //
  [249, [464, 177]],  //
  [390, [460, 177]],  //
  [373, [455, 178]],  //
  [374, [448, 179]],  //
  [380, [441, 179]],  //
  [381, [435, 178]],  //
  [382, [432, 177]],  //
  [362, [430, 177]],  //
  [466, [465, 175]],  //
  [388, [462, 173]],  //
  [387, [457, 171]],  //
  [386, [450, 170]],  //
  [385, [444, 171]],  //
  [384, [437, 172]],  //
  [398, [432, 175]]   //
];

const IRIS_INDICES_TO_LANDMARKS: Array<[number, [number, number]]> = [
  [468, [362, 175]],  //
  [469, [371, 175]],  //
  [470, [362, 167]],  //
  [471, [354, 175]],  //
  [472, [363, 182]],  //
  [473, [449, 174]],  //
  [474, [458, 174]],  //
  [475, [449, 167]],  //
  [476, [440, 174]],  //
  [477, [449, 181]]   //
];

const EXPECTED_BOX: BoundingBox = {
  xMin: 305,
  xMax: 504,
  yMin: 103,
  yMax: 347,
  width: 199,
  height: 244
};

export async function expectFaceMesh(
    detector: faceLandmarksDetection.FaceLandmarksDetector,
    image: HTMLImageElement, staticImageMode: boolean, refineLandmarks: boolean,
    numFrames: number, epsilon: number) {
  for (let i = 0; i < numFrames; ++i) {
    const result = await detector.estimateFaces(image, {staticImageMode});
    expect(result.length).toBe(1);

    const box = result[0].box;
    expectNumbersClose(box.xMin, EXPECTED_BOX.xMin, EPSILON_IMAGE);
    expectNumbersClose(box.xMax, EXPECTED_BOX.xMax, EPSILON_IMAGE);
    expectNumbersClose(box.yMin, EXPECTED_BOX.yMin, EPSILON_IMAGE);
    expectNumbersClose(box.yMax, EXPECTED_BOX.yMax, EPSILON_IMAGE);
    expectNumbersClose(box.width, EXPECTED_BOX.width, EPSILON_IMAGE);
    expectNumbersClose(box.height, EXPECTED_BOX.height, EPSILON_IMAGE);

    const keypoints =
        result[0].keypoints.map(keypoint => [keypoint.x, keypoint.y]);
    expect(keypoints.length)
        .toBe(
            refineLandmarks ? MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS_WITH_IRISES :
                              MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS);

    for (const [eyeIdx, gtLds] of EYE_INDICES_TO_LANDMARKS) {
      expectArraysClose(keypoints[eyeIdx], gtLds, epsilon);
    }

    if (refineLandmarks) {
      for (const [irisIdx, gtLds] of IRIS_INDICES_TO_LANDMARKS) {
        expectArraysClose(keypoints[irisIdx], gtLds, epsilon);
      }
    }
  }
}

describeWithFlags('MediaPipe FaceMesh ', BROWSER_ENVS, () => {
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

  async function expectMediaPipeFaceMesh(
      image: HTMLImageElement, staticImageMode: boolean,
      refineLandmarks: boolean, numFrames: number) {
    // Note: this makes a network request for model assets.
    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detector = await faceLandmarksDetection.createDetector(
        model, {...MEDIAPIPE_MODEL_CONFIG, refineLandmarks});

    await expectFaceMesh(
        detector, image, staticImageMode, refineLandmarks, numFrames,
        EPSILON_IMAGE);
  }

  it('static image mode no attention.', async () => {
    await expectMediaPipeFaceMesh(image, true, false, 5);
  });

  it('static image mode with attention.', async () => {
    await expectMediaPipeFaceMesh(image, true, true, 5);
  });

  it('streaming mode no attention.', async () => {
    await expectMediaPipeFaceMesh(image, false, false, 10);
  });

  it('streaming mode with attention.', async () => {
    await expectMediaPipeFaceMesh(image, false, true, 10);
  });
});
