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

import * as mobilenet from '@tensorflow-models/mobilenet';

import {TaskModelLoader} from '../../task_model';
import {Class, ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';

import {ImageClassificationResult, ImageClassifier} from './common';

// The global namespace type.
type MobilenetNS = typeof mobilenet;

/** Loading options. */
export interface MobilenetTFJSLoadingOptions extends
    TFJSModelCommonLoadingOption, mobilenet.ModelConfig {}

/** Inference options. */
export interface MobilenetTFJSInferenceOptions {
  /** Number of top classes to return. */
  topK?: number;
}

/** Loader for mobilenet TFJS model. */
export class MobilenetTFJSLoader extends
    TaskModelLoader<MobilenetNS, MobilenetTFJSLoadingOptions, MobilenetTFJS> {
  readonly metadata = {
    name: 'TFJS Mobilenet',
    description: 'Run mobilenet image classification model with TFJS',
    resourceUrls: {
      'github':
          'https://github.com/tensorflow/tfjs-models/tree/master/mobilenet',
    },
    runtime: Runtime.TFJS,
    version: '2.1.0',
    supportedTasks: [Task.IMAGE_CLASSIFICATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'mobilenet';

  protected async transformSourceModel(
      sourceModelGlobal: MobilenetNS,
      loadingOptions?: MobilenetTFJSLoadingOptions): Promise<MobilenetTFJS> {
    const mobilenetModel = await sourceModelGlobal.load(loadingOptions);
    return new MobilenetTFJS(mobilenetModel, loadingOptions);
  }
}

/**
 * Pre-trained TFJS mobilenet model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * //
 * // By default, it uses mobilenet V1 with webgl backend. You can change them
 * // in the options parameter of the `load` function (see below for docs).
 * const model = await tfTask.ImageClassification.Mobilenet.TFJS.load();
 *
 * // Run inference on an image with options (optional).
 * const img = document.querySelector('img');
 * const result = await model.predict(img, {topK: 5});
 * console.log(result.classes);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.ImageClassifier` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol: 'MobilenetTFJSLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'MobilenetTFJSInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Image Classification', subheading: 'Models'}
 */
export class MobilenetTFJS extends
    ImageClassifier<MobilenetTFJSInferenceOptions> {
  constructor(
      private mobilenetModel?: mobilenet.MobileNet,
      private loadingOptions?: MobilenetTFJSLoadingOptions) {
    super();
  }

  async predict(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      infereceOptions?: MobilenetTFJSInferenceOptions):
      Promise<ImageClassificationResult> {
    if (!this.mobilenetModel) {
      throw new Error('source model is not loaded');
    }
    await ensureTFJSBackend(this.loadingOptions);
    const mobilenetResults = await this.mobilenetModel.classify(
        img, infereceOptions ? infereceOptions.topK : undefined);
    const classes: Class[] = mobilenetResults.map(result => {
      return {
        className: result.className,
        score: result.probability,
      };
    });
    const finalResult: ImageClassificationResult = {
      classes,
    };
    return finalResult;
  }
}

export const mobilenetTfjsLoader = new MobilenetTFJSLoader();
