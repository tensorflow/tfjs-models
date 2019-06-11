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
import { config } from './config';
import { Color, DeepLabInput, SemanticSegmentationBaseModel } from './types';

const pascalColormapMaxEntriesNum = config['DATASET_MAX_ENTRIES']['PASCAL'];
const createPascalColormap = (): Color[] => {
  /**
   * Generates the colormap matching the Pascal VOC dev guidelines.
   * The original implementation in Python: https://git.io/fjgw5
   */

  const colormap = new Array(pascalColormapMaxEntriesNum);
  for (let idx = 0; idx < pascalColormapMaxEntriesNum; ++idx) {
    colormap[idx] = new Array(3);
  }
  for (let shift = 7; shift > 4; --shift) {
    const indexShift = 3 * (7 - shift);
    for (let channel = 0; channel < 3; ++channel) {
      for (let idx = 0; idx < pascalColormapMaxEntriesNum; ++idx) {
        colormap[idx][channel] |=
          ((idx >> (channel + indexShift)) & 1) << shift;
      }
    }
  }
  return colormap;
};

const createADE20KColormap = (): Color[] => {
  return config['COLORMAPS']['ADE20K'] as Color[];
};

const createCityscapesColormap = (): Color[] => {
  return config['COLORMAPS']['CITYSCAPES'] as Color[];
};

export const getColormap = (base: SemanticSegmentationBaseModel) => {
  if (base === 'pascal') {
    return createPascalColormap();
  } else if (base === 'ade20k') {
    return createADE20KColormap();
  } else if (base === 'cityscapes') {
    return createCityscapesColormap();
  }
  throw new Error(
    `SemanticSegmentation cannot be constructed ` +
      `with an invalid base model ${base}. ` +
      `Try one of 'pascal', 'cityscapes' and 'ade20k'.`
  );
};

export const translateLabels = (base: SemanticSegmentationBaseModel) => {
  if (base === 'pascal') {
    return config['LABELS']['PASCAL'];
  } else if (base === 'ade20k') {
    return config['LABELS']['ADE20K'];
  } else if (base === 'cityscapes') {
    return config['LABELS']['CITYSCAPES'];
  }
  throw new Error(
    `SemanticSegmentation cannot be constructed ` +
      `with an invalid base model ${base}. ` +
      `Try one of 'pascal', 'cityscapes' and 'ade20k'.`
  );
};

export function toInputTensor(input: DeepLabInput) {
  return tf.tidy(() => {
    const image =
      input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
    const [height, width] = image.shape;
    const resizeRatio = config['CROP_SIZE'] / Math.max(width, height);
    const targetSize = [height, width].map(side =>
      Math.round(side * resizeRatio)
    );
    return tf.image
      .resizeBilinear(image, targetSize as [number, number])
      .expandDims(0);
  });
}
