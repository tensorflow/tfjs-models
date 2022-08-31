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

import * as tflite from '@tensorflow/tfjs-tflite';
import {TaskModelLoader} from '../../task_model';
import {Runtime, Task} from '../common';
import {ImageClassifierTFLite} from './tflite_common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export interface MobilenetTFLiteLoadingOptions extends
    tflite.ImageClassifierOptions {
  /**
   * The MobileNet version number. Use 1 for MobileNetV1, and 2 for
   * MobileNetV2. Defaults to 1.
   */
  version?: 1|2;
  /**
   * Controls the width of the network, trading accuracy for performance. A
   * smaller alpha decreases accuracy and increases performance. Defaults
   * to 1.0.
   */
  alpha?: 0.25|0.50|0.75|1.0;
}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface MobilenetTFLiteInferenceOptions {}

/** Loader for mobilenet TFLite model. */
export class MobilenetTFLiteLoader extends
    TaskModelLoader<TFLiteNS, MobilenetTFLiteLoadingOptions, MobilenetTFLite> {
  readonly metadata = {
    name: 'TFLite Mobilenet',
    description: 'Run mobilenet image classification model with TFLite',
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.IMAGE_CLASSIFICATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: MobilenetTFLiteLoadingOptions):
      Promise<MobilenetTFLite> {
    // Construct the mobilenet model url from version and alpha.
    let mobilenetVersion = '1';
    let mobilenetAlpha = '1.0';
    if (loadingOptions) {
      if (loadingOptions.version !== undefined) {
        mobilenetVersion = String(loadingOptions.version);
      }
      if (loadingOptions.alpha !== undefined) {
        mobilenetAlpha = String(loadingOptions.alpha);
      }
    }
    // There is no TFLite mobilenet v2 model available other than 2_1.0.
    if (mobilenetVersion === '2' && mobilenetAlpha !== '1.0') {
      mobilenetAlpha = '1.0';
      console.warn(`No mobilenet TFLite model available for ${
          mobilenetVersion}_${mobilenetAlpha}. Using 2_1.0 instead.`);
    }
    // TODO: use TFHub url when CORS is correctly set.
    const url = `https://storage.googleapis.com/tfweb/models/mobilenet_v${
        mobilenetVersion}_${mobilenetAlpha}_224_1_metadata_1.tflite`;
    const tfliteImageClassifier =
        await sourceModelGlobal.ImageClassifier.create(url, loadingOptions);
    return new MobilenetTFLite(tfliteImageClassifier);
  }
}

/**
 * Pre-trained TFLite mobilenet image classification model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * //
 * // By default, it uses mobilenet V1. You can change it in the options
 * // parameter of the `load` function (see below for docs).
 * const model = await tfTask.ImageClassification.Mobilenet.TFJS.load();
 *
 * // Run inference on an image.
 * const img = document.querySelector('img');
 * const result = await model.predict(img);
 * console.log(result.classes);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.ImageClassifier` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol:
 * 'MobilenetTFLiteLoadingOptions'},
 *   {description: 'Options for `predict`',
 * symbol: 'MobilenetTFLiteInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Image Classification', subheading: 'Models'}
 */
export class MobilenetTFLite extends
    ImageClassifierTFLite<MobilenetTFLiteInferenceOptions> {}

export const mobilenetTfliteLoader = new MobilenetTFLiteLoader();
