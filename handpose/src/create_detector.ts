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

import {HandDetector} from './hand_detector';
import {load as loadMPHandsMediaPipeDetector} from './mediapipe/detector';
import {MPHandsMediaPipeModelConfig, MPHandsModelConfig} from './mediapipe/types';
import {SupportedModels} from './types';

/**
 * Create a hand detector instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
export async function createDetector(
    model: SupportedModels,
    modelConfig?: MPHandsMediaPipeModelConfig): Promise<HandDetector> {
  switch (model) {
    case SupportedModels.MPHands:
      const config = modelConfig as MPHandsModelConfig;
      let runtime;
      if (config != null) {
        if (config.runtime === 'tfjs') {
          throw new Error(
              `tfjs runtime does not yet support the updated mediapipe API.`);
        }
        if (config.runtime === 'mediapipe') {
          return loadMPHandsMediaPipeDetector(modelConfig);
        }
        runtime = config.runtime;
      }
      throw new Error(
          `Expect modelConfig.runtime to be either 'tfjs' ` +
          `or 'mediapipe', but got ${runtime}`);
    default:
      throw new Error(`${model} is not a supported model name.`);
  }
}
