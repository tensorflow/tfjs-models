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
import {Class} from '../common';

/**
 * The base class for all NLClassification task models.
 *
 * @template IO The type of options used during the inference process. Different
 *     models have different inference options. See individual model for more
 *     details.
 *
 * @doc {heading: 'NL Classification', subheading: 'Base model'}
 */
export abstract class NLClassifier<IO> implements TaskModel {
  /**
   * Predicts classes on the given text, and returns result.
   *
   * @param text The text to predict on.
   * @param options Inference options. Different models have different
   *     inference options. See individual model for more details.
   * @returns
   *
   * @docunpackreturn ['NLClassificationResult', 'Class']
   * @doc {heading: 'NL Classification', subheading: 'Base model'}
   */
  abstract predict(text: string, options?: IO): Promise<NLClassificationResult>;

  /**
   * Cleans up resources if needed.
   *
   * @doc {heading: 'NL Classification', subheading: 'Base model'}
   */
  cleanUp() {}
}

/** NL classification result. */
export interface NLClassificationResult {
  /** All predicted classes. */
  classes: Class[];
}
