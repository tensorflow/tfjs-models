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

const BASE_PATH = 'https://storage.googleapis.com/tfjs-testing/use-qa-2/';
const SKIP_VALUES = [0, 1, 2];
const OFFSET = 3;
const INPUT_LIMIT = 192;
const QUERY_NODE_NAME = 'input_inp_text';
const RESPONSE_CONTEXT_NODE_NAME = 'input_res_context';
const RESPONSE_NODE_NAME = 'input_res_text';
const RESPONSE_RESULT_NODE_NAME = 'Final/EncodeResult/mul';
const QUERY_RESULT_NODE_NAME = 'Final/EncodeQuery/mul';
const RESERVED_SYMBOLS_COUNT = 3;
export interface ModelOutput {
  queryEmbedding: number[][];
  responseEmbedding: number[][];
}

export interface Response {
  response: string;
  context?: string;
}
export interface ModelInput {
  queries: string[];
  responses: Response[];
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
    return tfconv.loadGraphModel(BASE_PATH + 'model.json');
  }

  async load() {
    const [model, vocabulary] = await Promise.all(
        [this.loadModel(), loadVocabulary(`${BASE_PATH}vocab.json`)]);

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
      const queryTokens = input.queries.map(
          q => this.updateToken(this.tokenizer.encode(q), INPUT_LIMIT));
      const queryEncoding = tf.tensor2d(
          queryTokens, [input.queries.length, INPUT_LIMIT], 'int32');
      const responseTokens = input.responses.map(
          r =>
              this.updateToken(this.tokenizer.encode(r.response), INPUT_LIMIT));
      const responseEncoding = tf.tensor2d(
          responseTokens, [input.responses.length, INPUT_LIMIT], 'int32');
      const contextTokens = input.responses.map(
          r => this.updateToken(
              this.tokenizer.encode(r.context || ''), INPUT_LIMIT));
      const contextEncoding = tf.tensor2d(
          contextTokens, [input.responses.length, INPUT_LIMIT], 'int32');
      const modelInputs: {[key: string]: tf.Tensor} = {};
      modelInputs[QUERY_NODE_NAME] = queryEncoding;
      modelInputs[RESPONSE_NODE_NAME] = responseEncoding;
      modelInputs[RESPONSE_CONTEXT_NODE_NAME] = contextEncoding;

      return this.model.execute(
          modelInputs, [QUERY_RESULT_NODE_NAME, RESPONSE_RESULT_NODE_NAME]);
    }) as tf.Tensor[];
    const queryEmbedding = embeddings[0].arraySync() as number[][];
    const responseEmbedding = embeddings[1].arraySync() as number[][];
    embeddings[0].dispose();
    embeddings[1].dispose();
    return {queryEmbedding, responseEmbedding};
  }

  private updateToken(token: number[], limit: number): number[] {
    token.unshift(1);
    for (let index = 0; index < limit; index++) {
      if (index >= token.length) {
        token[index] = 2;
      } else if (SKIP_VALUES.indexOf(token[index]) === -1) {
        token[index] += OFFSET;
      }
    }
    return token.slice(0, limit);
  }
}
