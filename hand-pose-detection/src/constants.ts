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

export const MEDIAPIPE_KEYPOINTS = [
  'wrist',
  'thumb_cmc',
  'thumb_mcp',
  'thumb_ip',
  'thumb_tip',
  'index_finger_mcp',
  'index_finger_pip',
  'index_finger_dip',
  'index_finger_tip',
  'middle_finger_mcp',
  'middle_finger_pip',
  'middle_finger_dip',
  'middle_finger_tip',
  'ring_finger_mcp',
  'ring_finger_pip',
  'ring_finger_dip',
  'ring_finger_tip',
  'pinky_finger_mcp',
  'pinky_finger_pip',
  'pinky_finger_dip',
  'pinky_finger_tip',
];

export const FINGER_LOOKUP_INDICES = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [0, 5, 6, 7, 8],
  middleFinger: [0, 9, 10, 11, 12],
  ringFinger: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20],
};

export const MEDIAPIPE_CONNECTED_KEYPOINTS_PAIRS = [
  [0, 1],   [1, 2],   [2, 3],  [3, 4],   [0, 5],   [5, 6],  [6, 7],
  [7, 8],   [0, 9],   [9, 10], [10, 11], [11, 12], [0, 13], [13, 14],
  [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
];
