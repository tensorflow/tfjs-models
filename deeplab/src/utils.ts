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

import * as tf from '@tensorflow/tfjs-core';

import {config} from './config';
import {Color, DeepLabInput, Label, Legend, ModelArchitecture, QuantizationBytes, SegmentationData} from './types';

export function createPascalColormap(): Color[] {
  /**
   * Generates the colormap matching the Pascal VOC dev guidelines.
   * The original implementation in Python: https://git.io/fjgw5
   */

  const pascalColormapMaxEntriesNum = config['DATASET_MAX_ENTRIES']['PASCAL'];
  const colormap = new Array(pascalColormapMaxEntriesNum);
  for (let idx = 0; idx < pascalColormapMaxEntriesNum; ++idx) {
    colormap[idx] = new Array(3);
  }
  for (let shift = 7; shift > 4; --shift) {
    const indexShift = 3 * (7 - shift);
    for (let channel = 0; channel < 3; ++channel) {
      for (let idx = 0; idx < pascalColormapMaxEntriesNum; ++idx) {
        colormap[idx][channel] |= ((idx >> (channel + indexShift)) & 1)
            << shift;
      }
    }
  }
  return colormap;
}

/**
 * Returns
 *
 * @param base  :: `ModelArchitecture`
 *
 * The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
 *
 * @param quantizationBytes (optional) :: `QuantizationBytes`
 *
 * The degree to which weights are quantized (either 1, 2 or 4).
 * Setting this attribute to 1 or 2 will load the model with int32 and
 * float32 compressed to 1 or 2 bytes respectively.
 * Set it to 4 to disable quantization.
 *
 * @return The URL of the TF.js model
 */
export function getURL(
    base: ModelArchitecture, quantizationBytes: QuantizationBytes) {
  const TFHUB_BASE = `${config['BASE_PATH']}`;
  const TFHUB_QUERY_PARAM = 'tfjs-format=file';

  const modelPath = quantizationBytes === 4 ?
      `${base}/1/default/1/model.json` :
      `${base}/1/quantized/${quantizationBytes}/1/model.json`;

  // Example of url that should be generated.
  // https://tfhub.dev/tensorflow/tfjs-model/deeplab/pascal/1/default/1/model.json?tfjs-format=file
  return `${TFHUB_BASE}/${modelPath}?${TFHUB_QUERY_PARAM}`;
}

/**
 * @param base  :: `ModelArchitecture`
 *
 * The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
 *
 * @return colormap :: `[number, number, number][]`
 *
 * The list of colors in RGB format, represented as arrays and corresponding
 * to labels.
 */
export function getColormap(base: ModelArchitecture): Color[] {
  if (base === 'pascal') {
    return config['COLORMAPS']['PASCAL'] as Color[];
  } else if (base === 'ade20k') {
    return config['COLORMAPS']['ADE20K'] as Color[];
  } else if (base === 'cityscapes') {
    return config['COLORMAPS']['CITYSCAPES'] as Color[];
  }
  throw new Error(
      `SemanticSegmentation cannot be constructed ` +
      `with an invalid base model ${base}. ` +
      `Try one of 'pascal', 'cityscapes' and 'ade20k'.`);
}

/**
 * @param base  :: `ModelArchitecture`
 *
 * The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
 *
 * @return labellingScheme :: `string[]`
 *
 * The list with verbal descriptions of labels
 */
export function getLabels(base: ModelArchitecture) {
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
      `Try one of 'pascal', 'cityscapes' and 'ade20k'.`);
}

/**
 * @param input  ::
 * `ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|tf.Tensor3D`
 *
 * The input image to prepare for segmentation.
 *
 * @return resizedInput :: `string[]`
 *
 * The input tensor to run through the model.
 */
export function toInputTensor(input: DeepLabInput) {
  return tf.tidy(() => {
    const image =
        input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
    const [height, width] = image.shape;
    const resizeRatio = config['CROP_SIZE'] / Math.max(width, height);
    const targetHeight = Math.round(height * resizeRatio);
    const targetWidth = Math.round(width * resizeRatio);
    return tf.expandDims(
      tf.image.resizeBilinear(image, [targetHeight, targetWidth]));
  });
}

/**
 * @param colormap :: `Color[]`
 *
 * The list of colors in RGB format, represented as arrays and corresponding
 * to labels.
 *
 * @param labellingScheme :: `string[]`
 *
 * The list with verbal descriptions of labels
 *
 * @param rawSegmentationMap :: `tf.Tensor2D`
 *
 * The segmentation map of the image
 *
 * @param canvas (optional) :: `HTMLCanvasElement`
 *
 * The canvas where to draw the output
 *
 * @returns A promise of a `DeepLabOutput` object, with four attributes:
 *
 * - **legend** :: `{ [name: string]: [number, number, number] }`
 *
 *   The legend is a dictionary of objects recognized in the image and their
 *   colors in RGB format.
 *
 * - **height** :: `number`
 *
 *   The height of the returned segmentation map
 *
 * - **width** :: `number`
 *
 *   The width of the returned segmentation map
 *
 * - **segmentationMap** :: `Uint8ClampedArray`
 *
 *   The colored segmentation map as `Uint8ClampedArray` which can be
 *   fed into `ImageData` and mapped to a canvas.
 */
export async function toSegmentationImage(
    colormap: Color[],
    labelNames: string[],
    rawSegmentationMap: tf.Tensor2D,
    canvas?: HTMLCanvasElement,
    ): Promise<SegmentationData> {
  if (colormap.length < labelNames.length) {
    throw new Error(
        'The colormap must be expansive enough to encode each label. ' +
        `Aborting, since the given colormap has length ${colormap.length}, ` +
        `but there are ${labelNames.length} labels.`);
  }
  const [height, width] = rawSegmentationMap.shape;
  const segmentationImageBuffer = tf.buffer([height, width, 3], 'int32');
  const mapData = await rawSegmentationMap.array();
  const labels = new Set<Label>();
  for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
    for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
      const label: Label = mapData[columnIndex][rowIndex];
      labels.add(label);
      segmentationImageBuffer.set(colormap[label][0], columnIndex, rowIndex, 0);
      segmentationImageBuffer.set(colormap[label][1], columnIndex, rowIndex, 1);
      segmentationImageBuffer.set(colormap[label][2], columnIndex, rowIndex, 2);
    }
  }

  const segmentationImageTensor =
      segmentationImageBuffer.toTensor() as tf.Tensor3D;

  const segmentationMap =
      await tf.browser.toPixels(segmentationImageTensor, canvas);

  tf.dispose(segmentationImageTensor);

  const legend: Legend = {};
  for (const label of Array.from(labels)) {
    legend[labelNames[label]] = colormap[label];
  }
  return {legend, segmentationMap};
}
