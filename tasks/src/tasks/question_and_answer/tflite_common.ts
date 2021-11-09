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
import {Answer, QuestionAnswerer, QuestionAnswerResult} from './common';

/**
 * The base class for all Q&A TFLite models.
 *
 * @template T The type of inference options.
 */
export class QuestionAnswererTFLite<T> extends QuestionAnswerer<T> {
  constructor(private tfliteQuestionAnswerer: tflite.BertQuestionAnswerer) {
    super();
  }

  async predict(question: string, context: string, infereceOptions?: T):
      Promise<QuestionAnswerResult> {
    if (!this.tfliteQuestionAnswerer) {
      throw new Error('source model is not loaded');
    }
    // In TFLite task library, context is the first parameter.
    const tfliteResults = this.tfliteQuestionAnswerer.answer(context, question);
    const answers: Answer[] = tfliteResults.map(result => {
      return {
        text: result.text,
        startIndex: result.pos.start,
        endIndex: result.pos.end,
        score: result.pos.logit,
      };
    });
    return {answers};
  }

  cleanUp() {
    if (!this.tfliteQuestionAnswerer) {
      throw new Error('source model is not loaded');
    }
    this.tfliteQuestionAnswerer.cleanUp();
  }
}
