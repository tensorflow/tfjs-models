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

import {FaceDetector} from './face_detector';
import {load as loadMediaPipeFaceDetectorMediaPipe} from './mediapipe/detector';
import {MediaPipeFaceDetectorMediaPipeModelConfig, MediaPipeFaceDetectorModelConfig} from './mediapipe/types';
import {load as loadMediaPipeFaceDetectorTfjs} from './tfjs/detector';
import {MediaPipeFaceDetectorTfjsModelConfig} from './tfjs/types';
import {SupportedModels} from './types';

/**
 * Create a face detector instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
export async function createDetector(
    model: SupportedModels,
    modelConfig?: MediaPipeFaceDetectorMediaPipeModelConfig|
    MediaPipeFaceDetectorTfjsModelConfig): Promise<FaceDetector> {
  switch (model) {
    case SupportedModels.MediaPipeFaceDetector: {
      const config = modelConfig as MediaPipeFaceDetectorModelConfig;
      let runtime;
      if (config != null) {
        if (config.runtime === 'tfjs') {
          return loadMediaPipeFaceDetectorTfjs(
              config as MediaPipeFaceDetectorTfjsModelConfig);
        }
        if (config.runtime === 'mediapipe') {
          return loadMediaPipeFaceDetectorMediaPipe(
              config as MediaPipeFaceDetectorMediaPipeModelConfig);
        }
        runtime = config.runtime;
      }
      throw new Error(
          `Expect modelConfig.runtime to be either 'tfjs' ` +
          `or 'mediapipe', but got ${runtime}`);
    }
    default:
      throw new Error(`${model} is not a supported model name.`);
  }
}
