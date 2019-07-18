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

import 'bulma/css/bulma.css';
import {load} from '@tensorflow-models/text-detection';

const state = {};
const textDetection = {};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
      Number(selector.options[selector.selectedIndex].text);
  state.quantizationBytes = quantizationBytes;
  textDetection[quantizationBytes] = load({quantizationBytes});
  const runner = document.getElementById('run');
  runner.onclick = async () => {
    toggleInvisible('output-card', true);
    await tf.nextFrame();
    await runTextDetection(base);
  };
  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);
  status('Initialised models, waiting for input...');
};

const setImage = (src) => {
  toggleInvisible('output-card', true);
  const image = document.getElementById('input-image');
  image.src = src;
  toggleInvisible('input-card', false);
  status('Waiting until the model is picked...');
};

const processImage = (file) => {
  if (!file.type.match('image.*')) {
    return;
  }
  const reader = new FileReader();
  reader.onload = (event) => {
    setImage(event.target.result);
  };
  reader.readAsDataURL(file);
};

const processImages = (event) => {
  const files = event.target.files;
  Array.from(files).forEach(processImage);
};

const displayBoxes = (textDetectionOutput) => {
  const canvas = document.getElementById('output-image');
  const input = document.getElementById('input-image');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(input, 0, 0);
  // toggleInvisible('output-card', false);
  // const segmentationMapData = new ImageData(segmentationMap, width, height);
  // canvas.style.width = '100%';
  // canvas.style.height = '100%';
  // canvas.width = width;
  // canvas.height = height;
  // ctx.putImageData(segmentationMapData, 0, 0);

  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const runPrediction = (input, initialisationStart) => {
  deeplab[state.quantizationBytes].then((model) => {
    model.predict(input).then((output) => {
      status(`Obtained ${JSON.stringify(output)}`);
      displayBoxes(output);
      status(`Ran in ${
        ((performance.now() - initialisationStart) / 1000).toFixed(2)} s`);
    });
  });
};

const runTextDetection = async () => {
  status(`Running the inference...`);
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
      Number(selector.options[selector.selectedIndex].text);
  if (state.quantizationBytes !== quantizationBytes) {
    if (textDetection[quantizationBytes]) {
      (await textDetection[quantizationBytes]).dispose();
      textDetection[quantizationBytes] = undefined;
    }
    state.quantizationBytes = quantizationBytes;
  }
  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }
  toggleInvisible('input-card', false);

  if (!textDetection[quantizationBytes]) {
    status('Loading the model...');
    const loadingStart = performance.now();
    textDetection[quantizationBytes] = load({quantizationBytes});
    await textDetection[quantizationBytes];
    status(`Loaded the model in ${
      ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
  }
  const predictionStart = performance.now();
  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(input, predictionStart);
  } else {
    input.onload = () => {
      runPrediction(input, predictionStart);
    };
  }
};

window.onload = initializeModels;
