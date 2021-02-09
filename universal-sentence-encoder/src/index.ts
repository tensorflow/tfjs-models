/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {loadTokenizer, loadVocabulary, Tokenizer} from './tokenizer';
import {loadQnA} from './use_qna';

export {version} from './version';

const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder';

declare interface ModelInputs extends tf.NamedTensorMap {
  indices: tf.Tensor;
  values: tf.Tensor;
}

interface LoadConfig {
  modelUrl?: string;
  vocabUrl?: string;
}

export async function load(config?: LoadConfig) {
  const use = new UniversalSentenceEncoder();
  await use.load(config);
  return use;
}

export class UniversalSentenceEncoder {
  private model: tfconv.GraphModel;
  private tokenizer: Tokenizer;

  async loadModel(modelUrl?: string) {
    return modelUrl
      ? tfconv.loadGraphModel(modelUrl)
      : tfconv.loadGraphModel(
          'https://tfhub.dev/tensorflow/tfjs-model/universal-sentence-encoder-lite/1/default/1',
          {fromTFHub: true}
        );
  }

  async load(config: LoadConfig = {}) {
    const [model, vocabulary] = await Promise.all([
      this.loadModel(config.modelUrl),
      loadVocabulary(config.vocabUrl || `${BASE_PATH}/vocab.json`)
    ]);

    this.model = model;
    this.tokenizer = new Tokenizer(vocabulary);
  }

  /**
   *
   * Returns a 2D Tensor of shape [input.length, 512] that contains the
   * Universal Sentence Encoder embeddings for each input.
   *
   * @param inputs A string or an array of strings to embed.
   */
  async embed(inputs: string[]|string): Promise<tf.Tensor2D> {
    if (typeof inputs === 'string') {
      inputs = [inputs];
    }

    const encodings = inputs.map(d => this.tokenizer.encode(d));

    const indicesArr =
        encodings.map((arr, i) => arr.map((d, index) => [i, index]));

    let flattenedIndicesArr: Array<[number, number]> = [];
    for (let i = 0; i < indicesArr.length; i++) {
      flattenedIndicesArr =
          flattenedIndicesArr.concat(indicesArr[i] as Array<[number, number]>);
    }

    const indices = tf.tensor2d(
        flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
    const values = tf.tensor1d(tf.util.flatten(encodings) as number[], 'int32');

    const modelInputs: ModelInputs = {indices, values};

    const embeddings = await this.model.executeAsync(modelInputs);
    indices.dispose();
    values.dispose();

    return embeddings as tf.Tensor2D;
  }
}

export {Tokenizer};
export {loadTokenizer};
export {loadQnA};
