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
import {BodyPixSegmentationConfig} from './body_pix/types';
import {MediaPipeSelfieSegmentationMediaPipeSegmentationConfig} from './selfie_segmentation_mediapipe/types';
import {MediaPipeSelfieSegmentationTfjsSegmentationConfig} from './selfie_segmentation_tfjs/types';
import {Segmentation} from './shared/calculators/interfaces/common_interfaces';
import {BodySegmenterInput} from './types';

/**
 * User-facing interface for all body segmenters.
 */
export interface BodySegmenter {
  /**
   * Segments people in the input image.
   *
   * @param input The image to segment. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param segmentationConfig common config for `segmentPeople`.
   */
  segmentPeople(
      input: BodySegmenterInput,
      segmentationConfig?:
          MediaPipeSelfieSegmentationMediaPipeSegmentationConfig|
      MediaPipeSelfieSegmentationTfjsSegmentationConfig|
      BodyPixSegmentationConfig): Promise<Segmentation[]>;

  /**
   * Dispose the underlying models from memory.
   */
  dispose(): void;

  /**
   * Reset global states in the model.
   */
  reset(): void;
}
