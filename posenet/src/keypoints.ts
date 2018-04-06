/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

export type Tuple<T> = [T, T];
export type StringTuple = Tuple<string>;
export type NumberTuple = Tuple<number>;

export const jointNames = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
  'right_ankle'
];

export const NUM_KEYPOINTS = jointNames.length;

export interface NumberDict { [jointName: string]: number; }

export const jointIdsByName =
    jointNames.reduce((result: NumberDict, jointName, i): NumberDict => {
      result[jointName] = i;
      return result;
    }, {}) as NumberDict;

const connectedJointNames: StringTuple[] = [
  ['left_hip', 'left_shoulder'], ['left_elbow', 'left_shoulder'],
  ['left_elbow', 'left_wrist'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['right_hip', 'right_shoulder'],
  ['right_elbow', 'right_shoulder'], ['right_elbow', 'right_wrist'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
  ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip']
];

export const connectedJointIndeces = connectedJointNames.map(
    ([jointNameA, jointNameB]) =>
        ([jointIdsByName[jointNameA], jointIdsByName[jointNameB]]));
