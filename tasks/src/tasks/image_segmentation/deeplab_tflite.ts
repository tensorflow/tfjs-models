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
import {ImageSegmenterTFLite} from './tflite_common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export interface DeeplabTFLiteLoadingOptions extends
    tflite.ImageSegmenterOptions {}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface DeeplabTFLiteInferenceOptions {}

/** Loader for deeplab TFLite model. */
export class DeeplabTFLiteLoader extends
    TaskModelLoader<TFLiteNS, DeeplabTFLiteLoadingOptions, DeeplabTFLite> {
  readonly metadata = {
    name: 'TFLite Deeplab',
    description: 'Run Deeplab image segmentation model with TFLite',
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.IMAGE_SEGMENTATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: DeeplabTFLiteLoadingOptions): Promise<DeeplabTFLite> {
    const url = 'https://tfhub.dev/tensorflow/lite-model/' +
        'deeplabv3/1/metadata/2?lite-format=tflite';
    const tfliteImageSegmenter =
        await sourceModelGlobal.ImageSegmenter.create(url, loadingOptions);
    return new DeeplabTFLite(tfliteImageSegmenter);
  }
}

/**
 * Pre-trained TFLite deeplab image segmentation model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * const model = await tfTask.ImageSegmentation.Deeplab.TFLite.load();
 *
 * // Run inference on an image.
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
 *   {description: 'Options for `load`', symbol:
 * 'DeeplabTFLiteLoadingOptions'},
 *   {description: 'Options for `predict`',
 * symbol: 'DeeplabTFLiteInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Image Segmentation', subheading: 'Models'}
 */
export class DeeplabTFLite extends
    ImageSegmenterTFLite<DeeplabTFLiteInferenceOptions> {}

export const deeplabTfliteLoader = new DeeplabTFLiteLoader();
