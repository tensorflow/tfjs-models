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

import {loadBlazePoseMediaPipeDetector} from './blazepose_mediapipe/detector';
import {BlazePoseMediaPipeModelConfig, BlazePoseModelConfig} from './blazepose_mediapipe/types';
import {loadBlazePoseTfjsDetector} from './blazepose_tfjs/detector';
import {BlazePoseTfjsModelConfig} from './blazepose_tfjs/types';
import {loadMoveNetDetector} from './movenet/detector';
import {MoveNetModelConfig} from './movenet/types';
import {PoseDetector} from './pose_detector';
import {loadPoseNetDetector} from './posenet/detector';
import {PosenetModelConfig} from './posenet/types';
import {SupportedModels} from './types';

/**
 * Create a pose detector instance.
 *
 * @param model The name of the pipeline to load.
 */
export async function createDetector(
    model: SupportedModels,
    modelConfig?: PosenetModelConfig|BlazePoseTfjsModelConfig|
    BlazePoseMediaPipeModelConfig|MoveNetModelConfig): Promise<PoseDetector> {
  switch (model) {
    case SupportedModels.PoseNet:
      return loadPoseNetDetector(modelConfig as PosenetModelConfig);
    case SupportedModels.BlazePose:
      const config = modelConfig as BlazePoseModelConfig;
      return config != null && config.runtime === 'tfjs' ?
          loadBlazePoseTfjsDetector(modelConfig as BlazePoseTfjsModelConfig) :
          loadBlazePoseMediaPipeDetector(
              modelConfig as BlazePoseMediaPipeModelConfig);
    case SupportedModels.MoveNet:
      return loadMoveNetDetector(modelConfig as MoveNetModelConfig);
    default:
      throw new Error(`${model} is not a supported model name.`);
  }
}
