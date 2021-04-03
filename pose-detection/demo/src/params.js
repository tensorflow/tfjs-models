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
import * as posedetection from '@tensorflow-models/pose-detection';

export const DEFAULT_LINE_WIDTH = 2;
export const DEFAULT_RADIUS = 4;

export const VIDEO_SIZE = {
  '640 X 480': {width: 640, height: 480},
  '640 X 360': {width: 640, height: 360}
};
export const STATE = {
  camera: {targetFPS: 60, sizeOption: '640 X 480'},
  model: {
    model: posedetection.SupportedModels.MoveNet,
  }
};
STATE.model[posedetection.SupportedModels.MediapipeBlazepose] = {
  scoreThreshold: 0.65
};
STATE.model[posedetection.SupportedModels.PoseNet] = {
  scoreThreshold: 0.5
};
STATE.model[posedetection.SupportedModels.MoveNet] = {
  modelType: 'Thunder',
  scoreThreshold: 0.3
};
