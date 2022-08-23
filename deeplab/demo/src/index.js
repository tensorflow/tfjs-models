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
import '@tensorflow/tfjs-backend-webgl';

import {load} from '@tensorflow-models/deeplab';
import * as tf from '@tensorflow/tfjs-core';

import ade20kExampleImage from './examples/ade20k.jpg';
import cityscapesExampleImage from './examples/cityscapes.jpg';
import pascalExampleImage from './examples/pascal.jpg';

const modelNames = ['pascal', 'cityscapes', 'ade20k'];
const deeplab = {};
const state = {};

const deeplabExampleImages = {
  pascal: pascalExampleImage,
  cityscapes: cityscapesExampleImage,
  ade20k: ade20kExampleImage,
};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  modelNames.forEach((base) => {
    const selector = document.getElementById('quantizationBytes');
    const quantizationBytes =
        Number(selector.options[selector.selectedIndex].text);
    state.quantizationBytes = quantizationBytes;
    deeplab[base] = load({base, quantizationBytes});
    const toggler = document.getElementById(`toggle-${base}-image`);
    toggler.onclick = () => setImage(deeplabExampleImages[base]);
    const runner = document.getElementById(`run-${base}`);
    runner.onclick = async () => {
      toggleInvisible('output-card', true);
      toggleInvisible('legend-card', true);
      await tf.nextFrame();
      await runDeeplab(base);
    };
  });
  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);
  status('Initialised models, waiting for input...');
};

const setImage = (src) => {
  toggleInvisible('output-card', true);
  toggleInvisible('legend-card', true);
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

const displaySegmentationMap = (modelName, deeplabOutput) => {
  const {legend, height, width, segmentationMap} = deeplabOutput;
  const canvas = document.getElementById('output-image');
  const ctx = canvas.getContext('2d');

  toggleInvisible('output-card', false);
  const segmentationMapData = new ImageData(segmentationMap, width, height);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.width = width;
  canvas.height = height;
  ctx.putImageData(segmentationMapData, 0, 0);

  const legendList = document.getElementById('legend');
  while (legendList.firstChild) {
    legendList.removeChild(legendList.firstChild);
  }

  Object.keys(legend).forEach((label) => {
    const tag = document.createElement('span');
    tag.innerHTML = label;
    const [red, green, blue] = legend[label];
    tag.classList.add('column');
    tag.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
    tag.style.padding = '1em';
    tag.style.margin = '1em';
    tag.style.color = '#ffffff';

    legendList.appendChild(tag);
  });
  toggleInvisible('legend-card', false);


  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const runPrediction = (modelName, input, initialisationStart) => {
  deeplab[modelName].then((model) => {
    model.segment(input).then((output) => {
      displaySegmentationMap(modelName, output);
      status(`Ran in ${
          ((performance.now() - initialisationStart) / 1000).toFixed(2)} s`);
    });
  });
};

const runDeeplab = async (modelName) => {
  status(`Running the inference...`);
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
      Number(selector.options[selector.selectedIndex].text);
  if (state.quantizationBytes !== quantizationBytes) {
    for (const base of modelNames) {
      if (deeplab[base]) {
        (await deeplab[base]).dispose();
        deeplab[base] = undefined;
      }
    };
    state.quantizationBytes = quantizationBytes;
  }
  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }
  toggleInvisible('input-card', false);

  if (!deeplab[modelName]) {
    status('Loading the model...');
    const loadingStart = performance.now();
    deeplab[modelName] = load({base: modelName, quantizationBytes});
    await deeplab[modelName];
    status(`Loaded the model in ${
        ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
  }
  const predictionStart = performance.now();
  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(modelName, input, predictionStart);
  } else {
    input.onload = () => {
      runPrediction(modelName, input, predictionStart);
    };
  }
};

window.onload = initializeModels;
