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
export const BLAZEPOSE_KEYPOINTS_BY_SIDE = {
  left: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
  right: [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
  middle: [0]
};
export const COCO_KEYPOINTS_BY_SIDE = {
  left: [1, 3, 5, 7, 9, 11, 13, 15],
  right: [2, 4, 6, 8, 10, 12, 14, 16],
  middle: [0]
};
