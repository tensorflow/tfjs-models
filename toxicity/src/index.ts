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

import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs';

import {padInputs} from './util';

const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/toxicity/';

export async function load() {
  const model = new ToxicityClassifier();
  await model.load();
  return model;
}

export class ToxicityClassifier {
  private tokenizer: use.Tokenizer;
  private model: tf.FrozenModel;
  private labels: string[];
  private threshold: number;
  private includeHeads: string[];

  constructor(threshold = 0.6, includeHeads: string[] = []) {
    this.threshold = threshold;
    this.includeHeads = includeHeads;
  }

  async loadModel() {
    return tf.loadFrozenModel(`${BASE_PATH}model.json`);
  }

  async loadTokenizer() {
    return use.loadTokenizer();
  }

  async load() {
    const [model, tokenizer] =
        await Promise.all([this.loadModel(), this.loadTokenizer()]);

    this.model = model;
    this.tokenizer = tokenizer;

    this.labels =
        model.outputs.map((d: {name: string}) => d.name.split('/')[0]);

    if (this.includeHeads.length === 0) {
      this.includeHeads = this.labels;
    }
  }

  async classify(inputs: string[]|string): Promise<Array<{
    label: string,
    results: Array<{probabilities: Float32Array, match: boolean}>
  }>> {
    if (typeof inputs === 'string') {
      inputs = [inputs];
    }

    // const encodings = padInputs(inputs.map(d => this.tokenizer.encode(d)));
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

    const labels = await this.model.executeAsync(
        {Placeholder_1: indices, Placeholder: values});

    indices.dispose();
    values.dispose();

    return (labels as Array<tf.Tensor2D>)
        .filter(
            (d, i) => this.includeHeads.find(label => label === this.labels[i]))
        .map((d: tf.Tensor2D, i: number) => {
          const data = d.dataSync() as Float32Array;
          const results = [];
          for (let input = 0; input < inputs.length; input++) {
            const probabilities = data.slice(input * 2, input * 2 + 2);
            let match = null;

            if (Math.max(probabilities[0], probabilities[1]) > this.threshold) {
              match = probabilities[0] < probabilities[1]
            }

            results.push({probabilities, match});
          }

          return {
            label: this.labels[i], results
          }
        });
  }
}