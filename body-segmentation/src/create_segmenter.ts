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

import {BodySegmenter} from './body_segmenter';
import {load as loadMediaPipeSelfieSegmentationMediaPipeDetector} from './selfie_segmentation_mediapipe/segmenter';
import {MediaPipeSelfieSegmentationMediaPipeModelConfig, MediaPipeSelfieSegmentationModelConfig} from './selfie_segmentation_mediapipe/types';
import {SupportedModels} from './types';

/**
 * Create a body segmenter instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
export async function createSegmenter(
    model: SupportedModels,
    modelConfig?: MediaPipeSelfieSegmentationMediaPipeModelConfig):
    Promise<BodySegmenter> {
  switch (model) {
    case SupportedModels.MediaPipeSelfieSegmentation:
      const config = modelConfig as MediaPipeSelfieSegmentationModelConfig;
      let runtime;
      if (config != null) {
        if (config.runtime === 'tfjs') {
          throw new Error(`tfjs runtime not yet supported.`);
        }
        if (config.runtime === 'mediapipe') {
          return loadMediaPipeSelfieSegmentationMediaPipeDetector(
              config as MediaPipeSelfieSegmentationMediaPipeModelConfig);
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
