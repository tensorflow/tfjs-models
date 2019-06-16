/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import { EfficientNetInput, EfficientNetBaseModel } from './types';
import { config } from './config';

const normalize = (image: tf.Tensor3D) => {
  const [height, width] = image.shape;
  const imageData = image.arraySync() as number[][][];
  const meanRGB = config['MEAN_RGB'].map(depth => depth * 255);
  const stddevRGB = config['STDDEV_RGB'].map(depth => depth * 255);
  for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
    for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
      imageData[columnIndex][rowIndex] = imageData[columnIndex][rowIndex].map(
        (depth, channel) => (depth - meanRGB[channel]) / stddevRGB[channel]
      );
    }
  }
};

const cropAndResize = (
  base: EfficientNetBaseModel,
  input: EfficientNetInput
): tf.Tensor3D => {
  return tf.tidy(() => {
    const image: tf.Tensor4D = (input instanceof tf.Tensor
      ? input
      : tf.browser.fromPixels(input)
    ).expandDims(0);

    const [height, width] = image.shape;

    const imageSize = config['CROP_SIZE'][base];
    const cropPadding = config['CROP_PADDING'];
    const paddedCenterCropSize = Math.round(
      Math.min(width, height) * (imageSize / (imageSize + cropPadding))
    );
    const offsetHeight = Math.round((height - paddedCenterCropSize + 1) / 2);
    const offsetWidth = Math.round((width - paddedCenterCropSize + 1) / 2);

    return tf.image
      .cropAndResize(
        image,
        [
          [
            offsetHeight,
            offsetWidth,
            paddedCenterCropSize + offsetHeight,
            paddedCenterCropSize + offsetWidth,
          ],
        ],
        [0],
        [imageSize, imageSize]
      )
      .squeeze([0]);
  });
};

export const toInputTensor = async (
  base: EfficientNetBaseModel,
  input: EfficientNetInput
) => {
  return tf.tidy(() => {
    const croppedAndResizedImage = cropAndResize(base, input);
    return normalize(croppedAndResizedImage);
  });
};
