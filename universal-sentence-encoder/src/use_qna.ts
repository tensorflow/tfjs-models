/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {loadVocabulary, Tokenizer} from './tokenizer';

export {version} from './version';

const BASE_PATH =
    'https://tfhub.dev/google/tfjs-model/universal-sentence-encoder-qa-ondevice/1';
// Index in the vocab file that needs to be skipped.
const SKIP_VALUES = [0, 1, 2];
// Offset value for skipped vocab index.
const OFFSET = 3;
// Input tensor size limit.
const INPUT_LIMIT = 192;
// Model node name for query.
const QUERY_NODE_NAME = 'input_inp_text';
// Model node name for query.
const RESPONSE_CONTEXT_NODE_NAME = 'input_res_context';
// Model node name for response.
const RESPONSE_NODE_NAME = 'input_res_text';
// Model node name for response result.
const RESPONSE_RESULT_NODE_NAME = 'Final/EncodeResult/mul';
// Model node name for query result.
const QUERY_RESULT_NODE_NAME = 'Final/EncodeQuery/mul';
// Reserved symbol count for tokenizer.
const RESERVED_SYMBOLS_COUNT = 3;
// Value for token padding
const TOKEN_PADDING = 2;
// Start value for each token
const TOKEN_START_VALUE = 1;

export interface ModelOutput {
  queryEmbedding: tf.Tensor;
  responseEmbedding: tf.Tensor;
}

export interface ModelInput {
  queries: string[];
  responses: string[];
  contexts?: string[];
}

export async function loadQnA() {
  const use = new UniversalSentenceEncoderQnA();
  await use.load();
  return use;
}

export class UniversalSentenceEncoderQnA {
  private model: tfconv.GraphModel;
  private tokenizer: Tokenizer;

  async loadModel() {
    return tfconv.loadGraphModel(BASE_PATH, {fromTFHub: true});
  }

  async load() {
    const [model, vocabulary] = await Promise.all([
      this.loadModel(),
      loadVocabulary(`${BASE_PATH}/vocab.json?tfjs-format=file`)
    ]);

    this.model = model;
    this.tokenizer = new Tokenizer(vocabulary, RESERVED_SYMBOLS_COUNT);
  }

  /**
   *
   * Returns a map of queryEmbedding and responseEmbedding
   *
   * @param input the ModelInput that contains queries and answers.
   */
  embed(input: ModelInput): ModelOutput {
    const embeddings = tf.tidy(() => {
      const queryEncoding = this.tokenizeStrings(input.queries, INPUT_LIMIT);
      const responseEncoding =
          this.tokenizeStrings(input.responses, INPUT_LIMIT);
      if (input.contexts != null) {
        if (input.contexts.length !== input.responses.length) {
          throw new Error(
              'The length of response strings ' +
              'and context strings need to match.');
        }
      }
      const contexts: string[] = input.contexts || [];
      if (input.contexts == null) {
        contexts.length = input.responses.length;
        contexts.fill('');
      }
      const contextEncoding = this.tokenizeStrings(contexts, INPUT_LIMIT);
      const modelInputs: {[key: string]: tf.Tensor} = {};
      modelInputs[QUERY_NODE_NAME] = queryEncoding;
      modelInputs[RESPONSE_NODE_NAME] = responseEncoding;
      modelInputs[RESPONSE_CONTEXT_NODE_NAME] = contextEncoding;

      return this.model.execute(
          modelInputs, [QUERY_RESULT_NODE_NAME, RESPONSE_RESULT_NODE_NAME]);
    }) as tf.Tensor[];
    const queryEmbedding = embeddings[0];
    const responseEmbedding = embeddings[1];

    return {queryEmbedding, responseEmbedding};
  }

  private tokenizeStrings(strs: string[], limit: number): tf.Tensor2D {
    const tokens =
        strs.map(s => this.shiftTokens(this.tokenizer.encode(s), INPUT_LIMIT));
    return tf.tensor2d(tokens, [strs.length, INPUT_LIMIT], 'int32');
  }

  private shiftTokens(tokens: number[], limit: number): number[] {
    tokens.unshift(TOKEN_START_VALUE);
    for (let index = 0; index < limit; index++) {
      if (index >= tokens.length) {
        tokens[index] = TOKEN_PADDING;
      } else if (!SKIP_VALUES.includes(tokens[index])) {
        tokens[index] += OFFSET;
      }
    }
    return tokens.slice(0, limit);
  }
}
