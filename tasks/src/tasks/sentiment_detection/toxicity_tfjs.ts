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

import * as toxicity from '@tensorflow-models/toxicity';

import {TaskModelLoader} from '../../task_model';
import {ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';
import {Sentiment, SentimentDetectionBaseOptions, SentimentDetectionResult, SentimentDetector} from './common';

// The global namespace type.
type ToxicityNS = typeof toxicity;

/** Loading options. */
export interface ToxicityTFJSLoadingOptions extends
    TFJSModelCommonLoadingOption, SentimentDetectionBaseOptions {
  /**
   * An array of strings indicating which types of toxicity to detect. Labels
   * must be one of `toxicity` | `severe_toxicity` | `identity_attack` |
   * `insult` | `threat` | `sexual_explicit` | `obscene`. Defaults to all
   * labels.
   */
  toxicityLabels?: string[];
}

/** Inference options (placeholder). */
export interface ToxicityTFJSInferenceOptions {}

/** Loader for toxicity TFJS model. */
export class ToxicityTFJSLoader extends
    TaskModelLoader<ToxicityNS, ToxicityTFJSLoadingOptions, ToxicityTFJS> {
  readonly metadata = {
    name: 'TFJS Toxicity model',
    description: 'Detect whether text contains toxic content such as ' +
        'threatening language, insults, obscenities, identity-based hate, ' +
        'or sexually explicit language.',
    resourceUrls: {
      'github':
          'https://github.com/tensorflow/tfjs-models/tree/master/toxicity',
    },
    runtime: Runtime.TFJS,
    version: '1.2.2',
    supportedTasks: [Task.SENTIMENT_DETECTION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/toxicity@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'toxicity';

  protected async transformSourceModel(
      sourceModelGlobal: ToxicityNS,
      loadingOptions?: ToxicityTFJSLoadingOptions): Promise<ToxicityTFJS> {
    const toxicityModel = await sourceModelGlobal.load(
        loadingOptions && loadingOptions.threshold ? loadingOptions.threshold :
                                                     0.85,
        loadingOptions && loadingOptions.toxicityLabels ?
            loadingOptions.toxicityLabels :
            []);
    return new ToxicityTFJS(toxicityModel, loadingOptions);
  }
}

/**
 * Pre-trained TFJS toxicity model.
 *
 * It detects whether text contains toxic content such as threatening language,
 * insults, obscenities, identity-based hate, or sexually explicit language.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional. See below for docs).
 * const model = await tfTask.SentimentDetection.Toxicity.TFJS.load();
 *
 * // Run detection on text.
 * const result = await model.predict('You are stupid');
 * console.log(result.sentimentLabels);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * By default, the model returns the prediction results of the following
 * sentiment labels:
 *
 * - toxicity
 * - severe_toxicity
 * - identity_attack
 * - insult
 * - threat
 * - sexual_explicit
 * - obscene
 *
 * Refer to `tfTask.SentimentDetection` for the `predict` and `cleanUp` method,
 * and more details about the result interface.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol: 'ToxicityTFJSLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'ToxicityTFJSInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Sentiment Detection', subheading: 'Models'}
 */
export class ToxicityTFJS extends
    SentimentDetector<ToxicityTFJSInferenceOptions> {
  constructor(
      private toxicityModel?: toxicity.ToxicityClassifier,
      private loadingOptions?: ToxicityTFJSLoadingOptions) {
    super();
  }

  async predict(text: string, options?: ToxicityTFJSInferenceOptions):
      Promise<SentimentDetectionResult> {
    if (!this.toxicityModel) {
      throw new Error('source model is not loaded');
    }
    await ensureTFJSBackend(this.loadingOptions);
    const toxicityResults = await this.toxicityModel.classify(text);
    const sentimentLabels: {[label: string]: Sentiment} = {};
    for (const labelResult of toxicityResults) {
      sentimentLabels[labelResult.label] = {
        result: labelResult.results[0].match,
        probabilities: Array.from(labelResult.results[0].probabilities),
      };
    }
    return {sentimentLabels};
  }
}

export const toxicityTfjsLoader = new ToxicityTFJSLoader();
