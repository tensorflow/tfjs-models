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
import * as tf from '@tensorflow/tfjs-core';
import {ImageSize} from './interfaces/common_interfaces';

import {TensorsToSegmentationConfig} from './interfaces/config_interfaces';

/**
 * Converts a tensor from a segmentation model to an image mask.
 * @param segmentationTensor Output from segmentation model of shape (1, height,
 *     width, channels).
 * @param config Contains activation to apply.
 * @param outputSize Desired dimensions of output image mask.
 *
 * @returns Image mask.
 */
export function tensorsToSegmentation(
    segmentationTensor: tf.Tensor4D, config: TensorsToSegmentationConfig,
    outputSize?: ImageSize): tf.Tensor2D {
  return tf.tidy(() => {
    // Remove batch dimension.
    const $segmentationTensor =
        // tslint:disable-next-line: no-unnecessary-type-assertion
        tf.squeeze(segmentationTensor, [0]) as tf.Tensor3D;

    const tensorChannels = $segmentationTensor.shape[2];
    // Process mask tensor and apply activation function.
    if (tensorChannels === 1) {
      // Create initial working mask.
      let smallMaskMat = $segmentationTensor;
      switch (config.activation) {
        case 'none':
          break;
        case 'sigmoid':
          smallMaskMat = tf.sigmoid(smallMaskMat);
          break;
        case 'softmax':
          throw new Error('Softmax activation requires two channels.');
        default:
          throw new Error(`Activation not supported (${config.activation})`);
      }

      const outputMat = outputSize ?
          tf.image.resizeBilinear(
              smallMaskMat, [outputSize.height, outputSize.width]) :
          smallMaskMat;

      // Remove channel dimension.
      return tf.squeeze(outputMat, [2]);
    } else {
      throw new Error(
          `Unsupported number of tensor channels ${tensorChannels}`);
    }
  });
}
