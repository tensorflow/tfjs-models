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

import {load as loadBodyPixSegmenter} from './body_pix/segmenter';
import {BodyPixModelConfig} from './body_pix/types';
import {BodySegmenter} from './body_segmenter';
import {load as loadMediaPipeSelfieSegmentationMediaPipeSegmenter} from './selfie_segmentation_mediapipe/segmenter';
import {MediaPipeSelfieSegmentationMediaPipeModelConfig, MediaPipeSelfieSegmentationModelConfig} from './selfie_segmentation_mediapipe/types';
import {load as loadMediaPipeSelfieSegmentationTfjsSegmenter} from './selfie_segmentation_tfjs/segmenter';
import {MediaPipeSelfieSegmentationTfjsModelConfig} from './selfie_segmentation_tfjs/types';
import {SupportedModels} from './types';

/**
 * Create a body segmenter instance.
 *
 * @param model The name of the pipeline to load.
 * @param modelConfig The configuration for the pipeline to load.
 */
export async function createSegmenter(
    model: SupportedModels,
    modelConfig?: MediaPipeSelfieSegmentationMediaPipeModelConfig|
    MediaPipeSelfieSegmentationTfjsModelConfig|
    BodyPixModelConfig): Promise<BodySegmenter> {
  switch (model) {
    case SupportedModels.MediaPipeSelfieSegmentation: {
      const config = modelConfig as MediaPipeSelfieSegmentationModelConfig;
      let runtime;
      if (config != null) {
        if (config.runtime === 'tfjs') {
          return loadMediaPipeSelfieSegmentationTfjsSegmenter(
              config as MediaPipeSelfieSegmentationTfjsModelConfig);
        }
        if (config.runtime === 'mediapipe') {
          return loadMediaPipeSelfieSegmentationMediaPipeSegmenter(
              config as MediaPipeSelfieSegmentationMediaPipeModelConfig);
        }
        runtime = config.runtime;
      }
      throw new Error(
          `Expect modelConfig.runtime to be either 'tfjs' ` +
          `or 'mediapipe', but got ${runtime}`);
    }
    case SupportedModels.BodyPix: {
      const config = modelConfig as BodyPixModelConfig;
      return loadBodyPixSegmenter(config);
    }
    default:
      throw new Error(`${model} is not a supported model name.`);
  }
}
