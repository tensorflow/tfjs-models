/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import * as config from './vocab.json';

let defaultUsed = false;
export function createDefaultGPT2() {
    defaultUsed = true;
    console.log('VOCAB SIZE:', Object.keys(config.vocabulary).length);
    const vocabulary = new Map(Object.entries(config.vocabulary));
    const merges = config.merges;

    const preprocessor = new tfl.GPT2CausalLMPreprocessor({
      tokenizer: new tfl.GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 30
    });

    const backbone = new tfl.GPT2Backbone({
      vocabularySize: preprocessor.tokenizer.vocabularySize,
      numLayers: 4, // 12, // 4,
      numHeads: 4, // 12, // 4,
      hiddenDim: 64, // 768, // 8,
      intermediateDim: 128, // 3072, // 16,
      maxSequenceLength: preprocessor.packer.sequenceLength,
    });

    const gpt2 = new tfl.GPT2CausalLM({backbone, preprocessor});

    // const weights = {
    //   dummy: tf.tensor([1]),
    // };
    // gpt2.loadWeights(weights)
    return gpt2;
}

export class GPT2 {

  constructor(private gpt2?: tfl.GPT2CausalLM) {}

  // This api should be friendly to JS users.
  async generate(input: string): Promise<string> {
    console.log(`[INPUT]: '${input}'`);
    const inputTensor = tf.tensor([input, input]);
    const outputTensor = this.gpt2.generate(inputTensor);
    const output = (outputTensor.dataSync() as unknown as string[])[0];
    if (defaultUsed) {
      const cleaned = output.replace(/\u0120/g, ' ');
      console.log(`[OUTPUT]: '${cleaned}'`);
      return cleaned;
    }
    console.log(`[OUTPUT]: '${output}'`);
    return output;
  }
}
