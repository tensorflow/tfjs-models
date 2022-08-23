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

import * as qna from '@tensorflow-models/qna';

import {TaskModelLoader} from '../../task_model';
import {ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';
import {QuestionAnswerer, QuestionAnswerResult} from './common';

// The global namespace type.
type QnaNs = typeof qna;

/** Loading options. */
export interface BertQATFJSLoadingOptions extends TFJSModelCommonLoadingOption,
                                                  qna.ModelConfig {}

/**
 * Inference options.
 *
 * TODO: placeholder.
 */
export interface BertQATFJSInferenceOptions {}

/** Loader for Q&A TFJS model. */
export class BertQATFJSLoader extends
    TaskModelLoader<QnaNs, BertQATFJSLoadingOptions, BertQATFJS> {
  readonly metadata = {
    name: 'TFJS Bert Q&A model',
    description: 'Run Bert Q&A model with TFJS',
    resourceUrls: {
      'github': 'https://github.com/tensorflow/tfjs-models/tree/master/qna',
    },
    runtime: Runtime.TFJS,
    version: '1.0.0',
    supportedTasks: [Task.QUESTION_AND_ANSWER],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/qna@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'qna';

  protected async transformSourceModel(
      sourceModelGlobal: QnaNs,
      loadingOptions?: BertQATFJSLoadingOptions): Promise<BertQATFJS> {
    let modelConfig: qna.ModelConfig = null;
    if (loadingOptions && loadingOptions.modelUrl) {
      modelConfig = {modelUrl: loadingOptions.modelUrl};
    }
    if (loadingOptions && loadingOptions.fromTFHub != null && modelConfig) {
      modelConfig.fromTFHub = loadingOptions.fromTFHub;
    }
    const bertQaModel = await sourceModelGlobal.load(modelConfig);
    return new BertQATFJS(bertQaModel, loadingOptions);
  }
}

/**
 * Pre-trained TFJS Bert Q&A model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * const model = await tfTask.QuestionAndAnswer.BertQA.TFJS.load();
 *
 * // Run inference with question and context.
 * const result = await model.predict(question, context);
 * console.log(result.answers);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.QuestionAnswerer` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol: 'BertQATFJSLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'BertQATFJSInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Question & Answer', subheading: 'Models'}
 */
export class BertQATFJS extends QuestionAnswerer<BertQATFJSInferenceOptions> {
  constructor(
      private qnaModel?: qna.QuestionAndAnswer,
      private loadingOptions?: BertQATFJSLoadingOptions) {
    super();
  }

  async predict(
      question: string, context: string,
      infereceOptions?: BertQATFJSInferenceOptions):
      Promise<QuestionAnswerResult> {
    if (!this.qnaModel) {
      throw new Error('source model is not loaded');
    }
    await ensureTFJSBackend(this.loadingOptions);
    const qnaResults = await this.qnaModel.findAnswers(question, context);
    return {answers: qnaResults};
  }
}

export const bertQaTfjsLoader = new BertQATFJSLoader();
