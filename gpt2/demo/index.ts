/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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

import {GPT2, load} from '@tensorflow-models/gpt2';
import * as lil from 'lil-gui';
import * as tf from '@tensorflow/tfjs-core';
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';

setWasmPaths('node_modules/@tensorflow/tfjs-backend-wasm/wasm-out/');

(window as any).tf = tf;

tf.setBackend('webgl');

const state = {
  backend: tf.getBackend(),
};

const gui = new lil.GUI();
const backendController = gui.add(state, 'backend', ['wasm', 'webgl', 'webgpu', 'cpu'])
  .onChange(async (backend: string) => {
    const lastBackend = tf.getBackend();
    let success = false;
    try {
      success = await tf.setBackend(backend);
    } catch (e) {
      console.warn(e.message);
    }
    if (!success) {
      alert(`Failed to use backend ${backend}. Check the console for errors.`);
      tf.setBackend(lastBackend);
      state.backend = lastBackend;
      backendController.updateDisplay();
      return;
    }
  }).listen(true);

const textElement = document.querySelector(".model-textbox") as HTMLTextAreaElement;

function setText(text: string) {
  textElement.textContent = text;
}
function getText() {
  return textElement.textContent || '';
}

const button = document.querySelector('.generate-button') as HTMLButtonElement;
if (button == null) {
  throw new Error('No button found for generating text');
}

button.onclick = generate;

let gpt2: GPT2;
async function init() {
  gpt2 = await load();
  button.disabled = false;
}

async function generate() {
  button.disabled = true;

  const text = getText();
  setText(text + await gpt2.generate(text));
  button.disabled = false;
}

init();
