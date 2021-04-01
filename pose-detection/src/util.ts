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
import * as constants from './constants';
import {SupportedModels} from './types';

export function getKeypointIndexBySide(model: SupportedModels):
    {left: number[], right: number[], middle: number[]} {
  switch (model) {
    case SupportedModels.MediapipeBlazepose:
      return constants.BLAZEPOSE_KEYPOINTS_BY_SIDE;
    case SupportedModels.PoseNet:
      return constants.COCO_KEYPOINTS_BY_SIDE;
    case SupportedModels.MoveNet:
      return constants.COCO_KEYPOINTS_BY_SIDE;
    default:
      throw new Error(`Model ${model} is not supported.`);
  }
}
export function getAdjacentPairs(model: SupportedModels): number[][] {
  switch (model) {
    case SupportedModels.MediapipeBlazepose:
      return constants.BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS;
    case SupportedModels.PoseNet:
      return constants.COCO_CONNECTED_KEYPOINTS_PAIRS;
    case SupportedModels.MoveNet:
      return constants.COCO_CONNECTED_KEYPOINTS_PAIRS;
    default:
      throw new Error(`Model ${model} is not supported.`);
  }
}