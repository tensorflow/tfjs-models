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
 * The base class for all Q&A task models.
 *
 * @template IO The type of options used during the inference process. Different
 *     models have different inference options. See individual model for more
 *     details.
 *
 * @doc {heading: 'Question & Answer', subheading: 'Base model'}
 */
export abstract class QuestionAnswerer<IO> implements TaskModel {
  /**
   * Gets the answer to the given question based on the content of a given
   * passage.
   *
   * @param qeustion The question to find answer for.
   * @param context Context where the answer are looked up from.
   * @returns
   *
   * @docunpackreturn ['QuestionAnswerResult', 'Answer']
   * @doc {heading: 'Question & Answer', subheading: 'Base model'}
   */
  abstract predict(question: string, context: string, options?: IO):
      Promise<QuestionAnswerResult>;

  /**
   * Cleans up resources if needed.
   *
   * @doc {heading: 'Question & Answer', subheading: 'Base model'}
   */
  cleanUp() {}
}

/** Q&A result. */
export interface QuestionAnswerResult {
  /** All predicted answers. */
  answers: Answer[];
}

/** A single answer. */
export interface Answer {
  /** The text of the answer. */
  text: string;
  /** The index of the starting character of the answer in the passage. */
  startIndex: number;
  /** The index of the last character of the answer text. */
  endIndex: number;
  /** Indicates the confident level. */
  score: number;
}
