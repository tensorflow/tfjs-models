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
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle',
  'rightAnkle'
];

export const NUM_KEYPOINTS = jointNames.length;

export interface NumberDict {
  [jointName: string]: number;
}

export const jointIds =
    jointNames.reduce((result: NumberDict, jointName, i): NumberDict => {
      result[jointName] = i;
      return result;
    }, {}) as NumberDict;

const connectedJointNames: StringTuple[] = [
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
];

export const connectedJointIndeces = connectedJointNames.map(
    ([jointNameA, jointNameB]) =>
        ([jointIds[jointNameA], jointIds[jointNameB]]));
