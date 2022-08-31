/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as deeplab from '@tensorflow-models/deeplab';

import {TaskModelLoader} from '../../task_model';
import {ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';

import {ImageSegmentationResult, ImageSegmenter, Legend} from './common';

// The global namespace type.
type DeeplabNS = typeof deeplab;

/** Loading options. */
export interface DeeplabTFJSLoadingOptions extends TFJSModelCommonLoadingOption,
                                                   deeplab.ModelConfig {
  /** The backend to use to run TFJS models. Default to 'webgl'. */
  backend: 'cpu'|'webgl';
}

/** Inference options. */
export interface DeeplabTFJSInferenceOptions extends deeplab.PredictionConfig {}

/** Loader for deeplab TFJS model. */
export class DeeplapTFJSLoader extends
    TaskModelLoader<DeeplabNS, DeeplabTFJSLoadingOptions, DeeplabTFJS> {
  readonly metadata = {
    name: 'TFJS Deeplab',
    description: 'Run deeplab image segmentation model with TFJS',
    resourceUrls: {
      'github': 'https://github.com/tensorflow/tfjs-models/tree/master/deeplab',
    },
    runtime: Runtime.TFJS,
    version: '0.2.1',
    supportedTasks: [Task.IMAGE_SEGMENTATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/deeplab@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'deeplab';

  protected async transformSourceModel(
      sourceModelGlobal: DeeplabNS,
      loadingOptions?: DeeplabTFJSLoadingOptions): Promise<DeeplabTFJS> {
    const options: DeeplabTFJSLoadingOptions = {...loadingOptions} ||
        {backend: 'webgl'};
    if (options.base == null) {
      options.base = 'pascal';
    }
    if (options.quantizationBytes == null) {
      options.quantizationBytes = 2;
    }
    const deeplabModel = await sourceModelGlobal.load(options);
    return new DeeplabTFJS(deeplabModel, options);
  }
}

/**
 * Pre-trained TFJS depelab model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * //
 * // By default, it uses base='pascal' and quantizationBytes=2 with webgl
 * // backend. You can change them in the options parameter of the `load`
 * // function (see below for docs).
 * const model = await tfTask.ImageSegmentation.Deeplab.TFJS.load();
 *
 * // Run inference on an image with options (optional).
 * const img = document.querySelector('img');
 * const result = await model.predict(img);
 * console.log(result);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.ImageSegmenter` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol: 'DeeplabTFJSLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'DeeplabTFJSInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Image Segmentation', subheading: 'Models'}
 */
export class DeeplabTFJS extends ImageSegmenter<DeeplabTFJSInferenceOptions> {
  constructor(
      private deeplabModel?: deeplab.SemanticSegmentation,
      private loadingOptions?: DeeplabTFJSLoadingOptions) {
    super();
  }

  async predict(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      infereceOptions?: DeeplabTFJSInferenceOptions):
      Promise<ImageSegmentationResult> {
    if (!this.deeplabModel) {
      throw new Error('source model is not loaded');
    }
    await ensureTFJSBackend(this.loadingOptions);
    const deeplabResult = await this.deeplabModel.segment(img, infereceOptions);
    const legend: Legend = {};
    for (const name of Object.keys(deeplabResult.legend)) {
      const colors = deeplabResult.legend[name];
      legend[name] = {
        r: colors[0],
        g: colors[1],
        b: colors[2],
      };
    }
    return {
      legend,
      width: deeplabResult.width,
      height: deeplabResult.height,
      segmentationMap: deeplabResult.segmentationMap,
    };
  }

  cleanUp() {
    if (!this.deeplabModel) {
      throw new Error('source model is not loaded');
    }
    this.deeplabModel.dispose();
  }
}

export const deeplabTfjsLoader = new DeeplapTFJSLoader();
