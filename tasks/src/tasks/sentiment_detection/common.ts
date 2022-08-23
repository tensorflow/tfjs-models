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

import {TaskModel} from '../../task_model';

/**
 * The base class for all SentimentDetection task models.
 *
 * @template IO The type of options used during the inference process. Different
 *     models have different inference options. See individual model for more
 *     details.
 *
 * @doc {heading: 'Sentiment Detection', subheading: 'Base model'}
 */
export abstract class SentimentDetector<IO> implements TaskModel {
  /**
   * Detects sentiment on the given text, and returns result.
   *
   * @param text The text to detect sentiment on.
   * @param options Inference options. Different models have different
   *     inference options. See individual model for more details.
   * @returns
   *
   * @docunpackreturn ['SentimentDetectionResult', 'Sentiment']
   * @doc {heading: 'Sentiment Detection', subheading: 'Base model'}
   */
  abstract predict(text: string, options?: IO):
      Promise<SentimentDetectionResult>;

  /**
   * Cleans up resources if needed.
   *
   * @doc {heading: 'Sentiment Detection', subheading: 'Base model'}
   */
  cleanUp() {}
}

/** Sentiment detection result. */
export interface SentimentDetectionResult {
  /**
   * A map from sentiment labels to their detection result along with the raw
   * probabilities ([negative probability, positive probability]).
   *
   * For example:
   * {
   *   'insult': {result: true, probabilities: [0.3, 0.7]}
   *   'threat': {result: false, probabilities: [0.7, 0.3]}
   * }
   */
  sentimentLabels: {[label: string]: Sentiment};
}

/** Detailed detection result for a certain sentiment label. */
export interface Sentiment {
  /**
   * Whether the sentiment is considered true or false. It is set to null when
   * the result cannot be determined (e.g. below a threshold).
   */
  result: boolean|null;
  /** The raw probabilities for this sentiment.  */
  probabilities: number[];
}

export interface SentimentDetectionBaseOptions {
  /**
   * A prediction is considered valid only if its confidence exceeds the
   * threshold. Defaults to 0.65.
   */
  threshold?: number;
}
