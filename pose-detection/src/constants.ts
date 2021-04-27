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
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
  'right_ankle'
];
export const BLAZEPOSE_KEYPOINTS = [
  'nose',           'right_eye_inner', 'right_eye',      'right_eye_outer',
  'left_eye_inner', 'left_eye',        'left_eye_outer', 'right_ear',
  'left_ear',       'mouth_right',     'mouth_left',     'right_shoulder',
  'left_shoulder',  'right_elbow',     'left_elbow',     'right_wrist',
  'left_wrist',     'right_pinky',     'left_pinky',     'right_index',
  'left_index',     'right_thumb',     'left_thumb',     'right_hip',
  'left_hip',       'right_knee',      'left_knee',      'right_ankle',
  'left_ankle',     'right_heel',      'left_heel',      'right_foot_index',
  'left_foot_index'
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
export const COCO_CONNECTED_KEYPOINTS_PAIRS = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12],
  [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
];
export const BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS = [
  [0, 1],   [0, 4],   [1, 2],   [2, 3],   [3, 7],   [4, 5],
  [5, 6],   [6, 8],   [9, 10],  [11, 12], [11, 13], [11, 23],
  [12, 14], [14, 16], [12, 24], [13, 15], [15, 17], [16, 18],
  [16, 20], [15, 17], [15, 19], [15, 21], [16, 22], [17, 19],
  [18, 20], [23, 25], [23, 24], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [27, 31], [28, 32], [29, 31], [30, 32]
];
export const COCO_KEYPOINTS_BY_NAME: {[index: string]: number} = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16
};
export const BLAZEPOSE_KEYPOINTS_BY_NAME: {[index: string]: number} = {
  nose: 0,
  right_eye_inner: 1,
  right_eye: 2,
  right_eye_outer: 3,
  left_eye_inner: 4,
  left_eye: 5,
  left_eye_outer: 6,
  right_ear: 7,
  left_ear: 8,
  mouth_right: 9,
  mouth_left: 10,
  right_shoulder: 11,
  left_shoulder: 12,
  right_elbow: 13,
  left_elbow: 14,
  right_wrist: 15,
  left_wrist: 16,
  right_pinky: 17,
  left_pinky: 18,
  right_index: 19,
  left_index: 20,
  right_thumb: 21,
  left_thumb: 22,
  right_hip: 23,
  left_hip: 24,
  right_knee: 25,
  left_knee: 26,
  right_ankle: 27,
  left_ankle: 28,
  right_heel: 29,
  left_heel: 30,
  right_foot_index: 31,
  left_foot_index: 32
};
