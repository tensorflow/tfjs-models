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

import {load as loadBlazePoseMediaPipeDetector} from './blazepose_mediapipe/detector';
import {BlazePoseMediaPipeModelConfig, BlazePoseModelConfig} from './blazepose_mediapipe/types';
import {load as loadBlazePoseTfjsDetector} from './blazepose_tfjs/detector';
import {BlazePoseTfjsModelConfig} from './blazepose_tfjs/types';
import {load as loadMoveNetDetector} from './movenet/detector';
import {MoveNetModelConfig} from './movenet/types';
import {PoseDetector} from './pose_detector';
import {load as loadPoseNetDetector} from './posenet/detector';
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
      let runtime;
      if (config != null) {
        if (config.runtime === 'tfjs') {
          return loadBlazePoseTfjsDetector(
              modelConfig as BlazePoseTfjsModelConfig);
        }
        if (config.runtime === 'mediapipe') {
          return loadBlazePoseMediaPipeDetector(
              modelConfig as BlazePoseMediaPipeModelConfig);
        }
        runtime = config.runtime;
      }
      throw new Error(
          `Expect modelConfig.runtime to be either 'tfjs' ` +
          `or 'mediapipe', but got ${runtime}`);
    case SupportedModels.MoveNet:
      return loadMoveNetDetector(modelConfig as MoveNetModelConfig);
    default:
      throw new Error(`${model} is not a supported model name.`);
  }
}
