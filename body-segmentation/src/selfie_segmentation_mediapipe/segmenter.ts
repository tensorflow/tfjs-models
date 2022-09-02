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
import * as selfieSegmentation from '@mediapipe/selfie_segmentation';
import * as tf from '@tensorflow/tfjs-core';

import {BodySegmenter} from '../body_segmenter';
import {Mask, Segmentation} from '../shared/calculators/interfaces/common_interfaces';
import {assertMaskValue, toImageDataLossy, toTensorLossy} from '../shared/calculators/mask_util';
import {BodySegmenterInput} from '../types';

import {validateModelConfig} from './segmenter_utils';
import {MediaPipeSelfieSegmentationMediaPipeModelConfig, MediaPipeSelfieSegmentationMediaPipeSegmentationConfig} from './types';

class MediaPipeSelfieSegmentationMediaPipeMask implements Mask {
  constructor(private mask: selfieSegmentation.GpuBuffer) {}

  async toCanvasImageSource() {
    return this.mask;
  }

  async toImageData() {
    return toImageDataLossy(this.mask);
  }

  async toTensor() {
    return toTensorLossy(this.mask);
  }

  getUnderlyingType() {
    return 'canvasimagesource' as const ;
  }
}

function maskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  return 'person';
}

/**
 * MediaPipe segmenter class.
 */
class MediaPipeSelfieSegmentationMediaPipeSegmenter implements BodySegmenter {
  private readonly selfieSegmentationSolution:
      selfieSegmentation.SelfieSegmentation;

  // This will be filled out by asynchronous calls to onResults. They will be
  // stable after `await send` is called on the selfie segmentation solution.
  private segmentation: Segmentation[];

  private selfieMode = false;

  // Should not be called outside.
  constructor(config: MediaPipeSelfieSegmentationMediaPipeModelConfig) {
    this.selfieSegmentationSolution =
        new selfieSegmentation.SelfieSegmentation({
          locateFile: (path, base) => {
            if (config.solutionPath) {
              const solutionPath = config.solutionPath.replace(/\/+$/, '');
              return `${solutionPath}/${path}`;
            }
            return `${base}/${path}`;
          }
        });
    let modelSelection: 0|1;
    switch (config.modelType) {
      case 'landscape':
        modelSelection = 1;
        break;
      case 'general':
      default:
        modelSelection = 0;
        break;
    }
    this.selfieSegmentationSolution.setOptions({
      modelSelection,
      selfieMode: this.selfieMode,
    });
    this.selfieSegmentationSolution.onResults((results) => {
      this.segmentation = [{
        maskValueToLabel,
        mask: new MediaPipeSelfieSegmentationMediaPipeMask(
            results.segmentationMask)
      }];
    });
  }

  /**
   * Segment people found in an image or video frame.
   *
   * It returns a single segmentation which contains all the detected people
   * in the input.
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of one `Segmentation`.
   */
  async segmentPeople(
      input: BodySegmenterInput,
      segmentationConfig?:
          MediaPipeSelfieSegmentationMediaPipeSegmentationConfig):
      Promise<Segmentation[]> {
    if (segmentationConfig && segmentationConfig.flipHorizontal &&
        (segmentationConfig.flipHorizontal !== this.selfieMode)) {
      this.selfieMode = segmentationConfig.flipHorizontal;
      this.selfieSegmentationSolution.setOptions({
        selfieMode: this.selfieMode,
      });
    }
    // Cast to GL TexImageSource types.
    input = input instanceof tf.Tensor ?
        new ImageData(
            await tf.browser.toPixels(input), input.shape[1], input.shape[0]) :
        input;
    await this.selfieSegmentationSolution.send(
        {image: input as selfieSegmentation.InputImage});
    return this.segmentation;
  }

  dispose() {
    this.selfieSegmentationSolution.close();
  }

  reset() {
    this.selfieSegmentationSolution.reset();
    this.segmentation = null;
    this.selfieMode = false;
  }

  initialize(): Promise<void> {
    return this.selfieSegmentationSolution.initialize();
  }
}

/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeSelfieSegmentation loading process. Please find more details of
 * each parameters in the documentation of the
 * `MediaPipeSelfieSegmentationMediaPipeModelConfig` interface.
 */
export async function load(
    modelConfig: MediaPipeSelfieSegmentationMediaPipeModelConfig):
    Promise<BodySegmenter> {
  const config = validateModelConfig(modelConfig);
  const segmenter = new MediaPipeSelfieSegmentationMediaPipeSegmenter(config);
  await segmenter.initialize();
  return segmenter;
}
