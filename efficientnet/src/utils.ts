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
  return tf.tidy(() => {
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
    tf.dispose(image);
    return tf.tensor3d(imageData);
  });
};

const cropAndResize = (
  base: EfficientNetBaseModel,
  input: EfficientNetInput
): tf.Tensor3D => {
  return tf.tidy(() => {
    const image: tf.Tensor3D =
      input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);

    const [height, width] = image.shape;

    const imageSize = config['CROP_SIZE'][base];
    const cropPadding = config['CROP_PADDING'];
    const paddedCenterCropSize = Math.round(
      Math.min(width, height) * (imageSize / (imageSize + cropPadding))
    );
    const offsetHeight = Math.round((height - paddedCenterCropSize + 1) / 2);
    const offsetWidth = Math.round((width - paddedCenterCropSize + 1) / 2);

    const processedImage: tf.Tensor3D = tf.image
      .cropAndResize(
        image.expandDims(0),
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
    tf.dispose(image);
    return processedImage;
  });
};

export const toInputTensor = (
  base: EfficientNetBaseModel,
  input: EfficientNetInput
) => {
  return tf.tidy(() => {
    return normalize(cropAndResize(base, input)).expandDims(0);
  });
};

// export async function getTopKClasses(logits, topK) {
//   const values = await logits.data();

//   const valuesAndIndices = [];
//   for (let i = 0; i < values.length; i++) {
//     valuesAndIndices.push({ value: values[i], index: i });
//   }
//   valuesAndIndices.sort((a, b) => {
//     return b.value - a.value;
//   });
//   const topkValues = new Float32Array(topK);
//   const topkIndices = new Int32Array(topK);
//   for (let i = 0; i < topK; i++) {
//     topkValues[i] = valuesAndIndices[i].value;
//     topkIndices[i] = valuesAndIndices[i].index;
//   }

//   const topClassesAndProbs = [];
//   for (let i = 0; i < topkIndices.length; i++) {
//     topClassesAndProbs.push({
//       className: IMAGENET_CLASSES[topkIndices[i]],
//       probability: topkValues[i],
//     });
//   }
//   return topClassesAndProbs;
// }
