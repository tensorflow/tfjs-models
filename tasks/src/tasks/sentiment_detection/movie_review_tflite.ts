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
import {getNLClassifierOptions} from '../nl_classification/tflite_common';
import {SentimentDetectionBaseOptions, SentimentDetectionResult, SentimentDetector} from './common';

// The global namespace type.
type TFLiteNS = typeof tflite;

const DEFAULT_THRESHOLD = 0.5;

/** Loading options. */
export interface MovieReviewTFLiteLoadingOptions extends
    SentimentDetectionBaseOptions {}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface MovieReviewTFLiteInferenceOptions {}

/** Loader for cocossd TFLite model. */
export class MovieReviewTFLiteLoader extends TaskModelLoader<
    TFLiteNS, MovieReviewTFLiteLoadingOptions, MovieReviewTFLite> {
  readonly metadata = {
    name: 'TFLite movie review model',
    description: 'Run a movie review model with TFLite and output ' +
        'the probabilities of whether the review is positive or negetive.',
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.SENTIMENT_DETECTION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: MovieReviewTFLiteLoadingOptions):
      Promise<MovieReviewTFLite> {
    const url = 'https://storage.googleapis.com/tfweb/models/' +
        'movie_review_sentiment_classification.tflite';
    const tfliteNLClassifier = await sourceModelGlobal.NLClassifier.create(
        url, getNLClassifierOptions());
    const threshold = loadingOptions && loadingOptions.threshold != null ?
        loadingOptions.threshold :
        DEFAULT_THRESHOLD;
    return new MovieReviewTFLite(tfliteNLClassifier, threshold);
  }
}

/**
 * Pre-trained TFLite movie review sentiment detection model.
 *
 * It detects whether the review text is positive or negetive.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * const model = await tfTask.SentimentDetection.MovieReview.TFLite.load();
 *
 * // Run inference on a review text.
 * const result = await model.predict('This is a great movie!');
 * console.log(result.sentimentLabels);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * The model returns the prediction results of the following sentiment labels:
 *
 * - positive
 * - negative
 *
 * Refer to `tfTask.SentimentDetector` for the `predict` and `cleanUp` method,
 * and more details about the result interface.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol:
 * 'MovieReviewTFLiteLoadingOptions'},
 *   {description: 'Options for `predict`',
 * symbol: 'MovieReviewTFLiteInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Sentiment Detection', subheading: 'Models'}
 */
export class MovieReviewTFLite extends
    SentimentDetector<MovieReviewTFLiteInferenceOptions> {
  constructor(
      private tfliteNLClassifier: tflite.NLClassifier,
      private threshold: number) {
    super();
  }

  async predict(text: string, options?: MovieReviewTFLiteInferenceOptions):
      Promise<SentimentDetectionResult> {
    if (!this.tfliteNLClassifier) {
      throw new Error('source model is not loaded');
    }
    const tfliteResults = this.tfliteNLClassifier.classify(text);
    if (!tfliteResults) {
      return {sentimentLabels: {}};
    }

    const negativeProbability = tfliteResults[0].probability;
    const positiveProbability = tfliteResults[1].probability;
    let positiveResult: boolean|null = null;
    let negativeResult: boolean|null = null;
    if (Math.max(negativeProbability, positiveProbability) > this.threshold) {
      positiveResult = positiveProbability > negativeProbability;
      negativeResult = positiveProbability < negativeProbability;
    }
    return {
      sentimentLabels: {
        'positive': {
          result: positiveResult,
          probabilities: [negativeProbability, positiveProbability],
        },
        'negative': {
          result: negativeResult,
          probabilities: [positiveProbability, negativeProbability],
        }
      }
    };
  }
}

export const movieReviewTfliteLoader = new MovieReviewTFLiteLoader();
