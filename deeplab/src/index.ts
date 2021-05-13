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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {DeepLabInput, DeepLabOutput, ModelArchitecture, ModelConfig, PredictionConfig} from './types';
import {getColormap, getLabels, getURL, toInputTensor, toSegmentationImage} from './utils';

export {version} from './version';
export {
  getColormap,
  getLabels,
  getURL,
  ModelConfig,
  PredictionConfig,
  toSegmentationImage
};

/**
 * Initializes the DeepLab model and returns a `SemanticSegmentation` object.
 *
 * @param input ::
 *     `ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement`
 *
 *  The input image to feed through the network.
 *
 * @param config :: `ModelConfig`
 *
 * The configuration for the model with any of the following attributes:
 *
 *   * quantizationBytes (optional) :: `QuantizationBytes`
 *
 *      The degree to which weights are quantized (either 1, 2 or 4).
 *      Setting this attribute to 1 or 2 will load the model with int32 and
 *      float32 compressed to 1 or 2 bytes respectively.
 *      Set it to 4 to disable quantization.
 *
 *   * base (optional) :: `ModelArchitecture`
 *
 *      The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
 *
 *   * modelUrl (optional) :: `string`
 *
 *      The URL from which to load the TF.js GraphModel JSON.
 *      Inferred from `base` and `quantizationBytes` if undefined.
 *
 * @return The initialized `SemanticSegmentation` object
 */
export async function load(
    modelConfig: ModelConfig = {
      base: 'pascal',
      quantizationBytes: 2
    },
) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js.` +
        ` If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if (modelConfig.base) {
    if (['pascal', 'cityscapes', 'ade20k'].indexOf(modelConfig.base) === -1) {
      throw new Error(
          `SemanticSegmentation cannot be constructed ` +
          `with an invalid base model ${modelConfig.base}. ` +
          `Try one of 'pascal', 'cityscapes' and 'ade20k'.`);
    }
    if ([1, 2, 4].indexOf(modelConfig.quantizationBytes) === -1) {
      throw new Error(`Only quantization to 1, 2 or 4 bytes is supported.`);
    }
  } else if (!modelConfig.modelUrl) {
    throw new Error(
        `SemanticSegmentation can be constructed either by passing ` +
        `the weights URL or one of the supported base model names from ` +
        `'pascal', 'cityscapes' and 'ade20k',` +
        `together with the degree of quantization (either 1, 2 or 4).` +
        `Aborting, since neither has been provided.`);
  }

  const graphModel = await tfconv.loadGraphModel(
      modelConfig.modelUrl ||
      getURL(modelConfig.base, modelConfig.quantizationBytes));
  const deeplab = new SemanticSegmentation(graphModel, modelConfig.base);
  return deeplab;
}

export class SemanticSegmentation {
  readonly model: tfconv.GraphModel;
  readonly base: ModelArchitecture;
  constructor(
      graphModel: tfconv.GraphModel,
      base?: ModelArchitecture,
  ) {
    this.model = graphModel;
    this.base = base;
  }

  /**
   * Segments an arbitrary image and generates a two-dimensional tensor with
   * class labels assigned to each cell of the grid overlayed on the image ( the
   * maximum number of cells on the side is fixed to 513).
   *
   * @param input ::
   *     `ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement`
   *
   * The input image to segment.
   *
   * @return rawSegmentationMap :: `tf.Tensor2D`
   *
   * The segmentation map of the image
   */
  public predict(input: DeepLabInput): tf.Tensor2D {
    return tf.tidy(() => {
      const data = tf.cast(toInputTensor(input), 'int32');
      return tf.squeeze(this.model.execute(data) as tf.Tensor);
    });
  }

  /**
   * Segments an arbitrary image and generates a two-dimensional tensor with
   * class labels assigned to each cell of the grid overlayed on the image ( the
   * maximum number of cells on the side is fixed to 513).
   *
   * @param image :: `ImageData | HTMLImageElement | HTMLCanvasElement |
   * HTMLVideoElement | tf.Tensor3D`;
   *
   *   The image to segment
   *
   * @param config (optional) The configuration object for the segmentation:
   *
   * - **config.canvas** (optional) :: `HTMLCanvasElement`
   *
   *   The canvas where to draw the output
   *
   * - **config.colormap** (optional) :: `[number, number, number][]`
   *
   *   The array of RGB colors corresponding to labels
   *
   * - **config.labels** (optional) :: `string[]`
   *
   *   The array of names corresponding to labels
   *
   *   By [default](./src/index.ts#L81), `colormap` and `labels` are set
   * according to the `base` model attribute passed during initialization.
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

  public async segment(input: DeepLabInput, config: PredictionConfig = {}):
      Promise<DeepLabOutput> {
    if (!((config.colormap && config.labels) || this.base)) {
      throw new Error(
          `Calling the 'segment' method requires either the 'base'` +
          ` attribute to be defined ` +
          `(e.g. 'pascal', 'cityscapes' or'ade20k'),` +
          ` or 'colormap' and 'labels' options to be set. ` +
          `Aborting, since neither has been provided.`);
    } else if (!(config.colormap && config.labels)) {
      config.colormap = getColormap(this.base);
      config.labels = getLabels(this.base);
    }

    const {colormap, labels, canvas} = config;
    const rawSegmentationMap = tf.tidy(() => this.predict(input));

    const [height, width] = rawSegmentationMap.shape;
    const {legend, segmentationMap} =
        await toSegmentationImage(colormap, labels, rawSegmentationMap, canvas);

    tf.dispose(rawSegmentationMap);

    return {legend, height, width, segmentationMap};
  }

  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
