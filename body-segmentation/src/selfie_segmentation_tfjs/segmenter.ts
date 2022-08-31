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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {Tensor3D} from '@tensorflow/tfjs-core';

import {BodySegmenter} from '../body_segmenter';
import {MediaPipeSelfieSegmentationModelType} from '../selfie_segmentation_mediapipe/types';
import {convertImageToTensor} from '../shared/calculators/convert_image_to_tensor';
import {getImageSize} from '../shared/calculators/image_utils';
import {Mask, Segmentation} from '../shared/calculators/interfaces/common_interfaces';
import {assertMaskValue, toHTMLCanvasElementLossy, toImageDataLossy} from '../shared/calculators/mask_util';
import {tensorsToSegmentation} from '../shared/calculators/tensors_to_segmentation';
import {BodySegmenterInput} from '../types';
import * as constants from './constants';

import {validateModelConfig, validateSegmentationConfig} from './segmenter_utils';
import {MediaPipeSelfieSegmentationTfjsModelConfig, MediaPipeSelfieSegmentationTfjsSegmentationConfig} from './types';

class MediaPipeSelfieSegmentationTfjsMask implements Mask {
  constructor(private mask: Tensor3D) {}

  async toCanvasImageSource() {
    return toHTMLCanvasElementLossy(this.mask);
  }

  async toImageData() {
    return toImageDataLossy(this.mask);
  }

  async toTensor() {
    return this.mask;
  }

  getUnderlyingType() {
    return 'tensor' as const ;
  }
}

function maskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  return 'person';
}

/**
 * MediaPipeSelfieSegmentation TFJS segmenter class.
 */
class MediaPipeSelfieSegmentationTfjsSegmenter implements BodySegmenter {
  constructor(
      private readonly modelType: MediaPipeSelfieSegmentationModelType,
      private readonly model: tfconv.GraphModel) {}

  /**
   * Segment people found in an image or video frame.
   *
   * It returns a single segmentation which contains all the detected people
   * in the input.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of one `Segmentation`.
   */
  // TF.js implementation of the mediapipe selfie segmentation pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/mediapipe/modules/elfie_segmentation/selfie_segmentation_cpu.pbtxt
  async segmentPeople(
      image: BodySegmenterInput,
      segmentationConfig: MediaPipeSelfieSegmentationTfjsSegmentationConfig):
      Promise<Segmentation[]> {
    segmentationConfig = validateSegmentationConfig(segmentationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    const rgbaMask = tf.tidy(() => {
      // SelfieSegmentationCpu: ImageToTensorCalculator.
      // Resizes the input image into a tensor with a dimension desired by the
      // model.
      const {imageTensor: imageValueShifted} = convertImageToTensor(
          image,
          this.modelType === 'general' ?
              constants.SELFIE_SEGMENTATION_IMAGE_TO_TENSOR_GENERAL_CONFIG :
              constants.SELFIE_SEGMENTATION_IMAGE_TO_TENSOR_LANDSCAPE_CONFIG);

      // SelfieSegmentationCpu: InferenceCalculator
      // The model returns a tensor with the following shape:
      // [1 (batch), 144, 256] or [1 (batch), 256, 256, 2] depending on
      // modelType.
      const segmentationTensor =
          // Slice activation output only.
          tf.slice(
              this.model.predict(imageValueShifted) as tf.Tensor4D,
              [0, 0, 0, 1], -1);

      // SelfieSegmentationCpu: ImagePropertiesCalculator
      // Retrieves the size of the input image.
      const imageSize = getImageSize(image);

      // SelfieSegmentationCpu: TensorsToSegmentationCalculator
      // Processes the output tensors into a segmentation mask that has the same
      // size as the input image into the graph.
      const maskImage = tensorsToSegmentation(
          segmentationTensor,
          constants.SELFIE_SEGMENTATION_TENSORS_TO_SEGMENTATION_CONFIG,
          imageSize);

      // Grayscale to RGBA
      // tslint:disable-next-line: no-unnecessary-type-assertion
      const mask3D = tf.expandDims(maskImage, 2) as tf.Tensor3D;
      const rgMask = tf.pad(mask3D, [[0, 0], [0, 0], [0, 1]]);
      return tf.mirrorPad(rgMask, [[0, 0], [0, 0], [0, 2]], 'symmetric');
    });

    return [{
      maskValueToLabel,
      mask: new MediaPipeSelfieSegmentationTfjsMask(rgbaMask)
    }];
  }

  dispose() {
    this.model.dispose();
  }

  reset() {}
}

/**
 * Loads the MediaPipeSelfieSegmentationTfjs model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeSelfieSegmentationTfjs loading process. Please find more details
 * of each parameters in the documentation of the
 * `MediaPipeSelfieSegmentationTfjsModelConfig` interface.
 */
export async function load(
    modelConfig: MediaPipeSelfieSegmentationTfjsModelConfig):
    Promise<BodySegmenter> {
  const config = validateModelConfig(modelConfig);

  const modelFromTFHub = typeof config.modelUrl === 'string' &&
      (config.modelUrl.indexOf('https://tfhub.dev') > -1);

  const model =
      await tfconv.loadGraphModel(config.modelUrl, {fromTFHub: modelFromTFHub});

  return new MediaPipeSelfieSegmentationTfjsSegmenter(config.modelType, model);
}
