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
import {QuestionAnswererTFLite} from './tflite_common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/**
 * Loading options.
 *
 * TODO: placeholder for now.
 */
export interface BertQATFLiteLoadingOptions {}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface BertQATFLiteInferenceOptions {}

/** Loader for Bert Q&A TFLite model. */
export class BertQATFLiteLoader extends
    TaskModelLoader<TFLiteNS, BertQATFLiteLoadingOptions, BertQATFLite> {
  readonly metadata = {
    name: 'TFLite Bert Q&A model',
    description: 'Run Bert Q&A model with TFLite',
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.QUESTION_AND_ANSWER],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: BertQATFLiteLoadingOptions): Promise<BertQATFLite> {
    const url = 'https://tfhub.dev/tensorflow/lite-model/' +
        'mobilebert/1/metadata/1?lite-format=tflite';
    const tfliteQa = await sourceModelGlobal.BertQuestionAnswerer.create(url);
    return new BertQATFLite(tfliteQa);
  }
}

/**
 * Pre-trained TFLite Bert Q&A model.
 *
 * Usage:
 *
 * ```js
 * // Load the model.
 * const model = await tfTask.QuestionAndAnswer.BertQA.TFLite.load();
 *
 * // Run inference on an image.
 * const result = await model.predict(question, context);
 * console.log(result);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.QuestionAnswerer` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol:
 * 'BertQATFLiteLoadingOptions'},
 *   {description: 'Options for `predict`',
 * symbol: 'BertQATFLiteInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Question & Answer', subheading: 'Models'}
 */
export class BertQATFLite extends
    QuestionAnswererTFLite<BertQATFLiteInferenceOptions> {}

export const bertQaTfliteLoader = new BertQATFLiteLoader();
