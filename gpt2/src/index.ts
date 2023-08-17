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
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import {GPT2, createDefaultGPT2} from './gpt2';
export {GPT2} from './gpt2';

// const MODEL_PATH = `http://orderique.c.googlers.com:8080/model_data/model.json`;
const MODEL_PATH = 'https://storage.googleapis.com/tfjs-testing/gpt2-temp/model.json';

// Note that while `tfjs-core` is availble here, we shouldn't import any backends.
// Let the user choose which backends they want in their bundle.
tf; tfl; createDefaultGPT2; MODEL_PATH; // Prevent it from complaining about unused variables

export async function load(): Promise<GPT2>{
  console.log('gpt2 loading...');
  const gpt2 = (await tfl.loadLayersModel(MODEL_PATH, {strict: true})) as tfl.GPT2CausalLM;
  // const gpt2 = createDefaultGPT2();
  console.log('gpt2 loaded.');
  return new GPT2(gpt2);
}
