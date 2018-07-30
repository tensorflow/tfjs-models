/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {decodeMultiplePoses} from './multiPose/decodeMultiplePoses';
// tslint:disable-next-line:max-line-length
import {decodeAndScaleSegmentationAndPartMap} from './partMap/decodePartMap';
import {load, PoseNet} from './posenet_model';
// tslint:disable-next-line:max-line-length
import {decodeSinglePose} from './singlePose/decodeSinglePose';

export {checkpoints} from './checkpoints';
// tslint:disable-next-line:max-line-length
export {partChannelIds, partChannels, partIds, partNames, poseChain} from './keypoints';
export {Keypoint, Pose} from './types';
// tslint:disable-next-line:max-line-length
export {getAdjacentKeyPoints, getBoundingBox, getBoundingBoxPoints} from './util';
export {
  decodeAndScaleSegmentationAndPartMap,
  decodeMultiplePoses,
  decodeSinglePose
};
export {load, PoseNet};
