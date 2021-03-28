/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
export const COCO_KEYPOINTS = [
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];
export const COCO_CONNECTED_KEYPOINTS_PAIRS = [
  [0, 1], [0, 2], [5, 7], [7, 9], [5, 11], [11, 13], [13, 15], [6, 8], [8, 10],
  [6, 12], [12, 14], [14, 16]
];
export const BLAZEPOSE_KEYPOINTS_UPPERBODY = [
  'nose',         'rightEyeInner', 'rightEye',     'rightEyeOuter',
  'leftEyeInner', 'leftEye',       'leftEyeOuter', 'rightEar',
  'leftEar',      'mouthRight',    'mouthLeft',    'rightShoulder',
  'leftShoulder', 'rightElbow',    'leftElbow',    'rightWrist',
  'leftWrist',    'rightPinky',    'leftPinky',    'rightIndex',
  'leftIndex',    'rightThumb',    'leftThumb',    'rightHip',
  'leftHip'
];
export const BLAZEPOSE_KEYPOINTS_FULLBODY = [
  ...BLAZEPOSE_KEYPOINTS_UPPERBODY, 'rightKnee', 'leftKnee', 'rightAnkle',
  'leftAnkle', 'rightHeel', 'leftHeel', 'rightFoot', 'leftFoot'
];
export const BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS = [
  [0, 1],   [0, 4],   [1, 2],   [2, 3],   [3, 7],   [4, 5],   [5, 6],
  [6, 8],   [9, 10],  [11, 12], [11, 13], [11, 23], [12, 14], [14, 16],
  [12, 24], [13, 15], [15, 17], [16, 18], [15, 21], [16, 22], [17, 19],
  [18, 20], [19, 21], [20, 22], [23, 25], [23, 24], [24, 26], [25, 27],
  [26, 28], [27, 29], [28, 30], [27, 31], [28, 32], [29, 31], [30, 32]
];
export const BLAZEPOSE_KEYPOINTS_BY_SIDE = {
  left: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
  right: [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
  middle: [0]
};
