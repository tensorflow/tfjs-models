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

import * as tf from '@tensorflow/tfjs';

import {Tokenizer} from './tokenizer';
import {flatten} from './util';

const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/';

export async function load() {
  const use = new UniversalSentenceEncoder();
  await use.load();
  return use;
}

export class UniversalSentenceEncoder {
  private model: tf.FrozenModel;
  private tokenizer: Tokenizer;

  async loadModel() {
    return await tf.loadFrozenModel(
        `${BASE_PATH}tensorflowjs_model.pb`,
        `${BASE_PATH}weights_manifest.json`);
  }

  async loadVocabulary() {
    const vocabulary = await fetch(`${BASE_PATH}vocab.json`);
    return await vocabulary.json();
  }

  async load() {
    const [model, vocabulary] =
        await Promise.all([this.loadModel(), this.loadVocabulary()]);

    this.model = model;
    this.tokenizer = new Tokenizer(vocabulary);
  }

  /**
   *
   * Returns a 2D Tensor of shape [input.length, 512] that contains the
   * Universal Sentence Encoder embeddings for each input.
   *
   * @param inputs An array of strings to embed.
   */
  async embed(inputs: string[]): Promise<tf.Tensor2D> {
    const encodings = inputs.map(d => this.tokenizer.encode(d));

    const indices =
        flatten(encodings.map((arr, i) => arr.map((d, index) => [i, index])));

    return await this.model.executeAsync({
      indices: tf.tensor2d(flatten(indices), [indices.length, 2], 'int32'),
      values: tf.tensor1d(flatten(encodings), 'int32')
    }) as tf.Tensor2D;
  }
}
