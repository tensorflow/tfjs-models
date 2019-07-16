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
import {EfficientNet} from '@tensorflow-models/efficientnet';
import * as tf from '@tensorflow/tfjs';

const state = {
  quantizationBytes: 2,
};

const efficientnet = {};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'].forEach((modelName) => {
    if (efficientnet[modelName]) {
      efficientnet[modelName].dispose();
    }
    efficientnet[modelName] =
        new EfficientNet(modelName, state.quantizationBytes);
    const runner = document.getElementById(`run-${modelName}`);
    runner.onclick = async () => {
      toggleInvisible('classification-card', true);
      await tf.nextFrame();
      await runEfficientNet(modelName);
    };
  });
  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);
  status('Initialised models, waiting for input...');
};

const setImage = (src) => {
  toggleInvisible('classification-card', true);
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

const probabilityToColor = (probability) => {
  const colors = [
    '#a50026',
    '#d73027',
    '#f46d43',
    '#fdae61',
    '#fee08b',
    '#ffffbf',
    '#d9ef8b',
    '#a6d96a',
    '#66bd63',
    '#1a9850',
    '#006837',
  ];
  return colors[Math.round(probability * (colors.length - 1))];
};

// based on  https://stackoverflow.com/a/35970186/3581829
function invertColor(hex, bw) {
  if (hex.indexOf('#') === 0) {
    hex = hex.slice(1);
  }
  // convert 3-digit hex to 6-digits.
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  if (hex.length !== 6) {
    throw new Error('Invalid HEX color.');
  }
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);

  // http://stackoverflow.com/a/3943023/112731
  return (r * 0.299 + g * 0.587 + b * 0.114) > 150 ? '#000000' : '#FFFFFF';
}

const displayClassification = (modelName, classification) => {
  toggleInvisible('classification-card', false);
  const classificationList = document.getElementById('classification');
  while (classificationList.firstChild) {
    classificationList.removeChild(classificationList.firstChild);
  }

  classification.forEach(({probability, className}) => {
    const tags = document.createElement('div');
    tags.classList.add('column', 'is-inline-flex');

    const probabilityTag = document.createElement('span');
    probabilityTag.classList.add('tag', 'is-dark');
    probabilityTag.innerText = probability.toFixed(4);
    probabilityTag.style.height = 'auto';
    probabilityTag.style.borderRadius = '0';

    const classNameTag = document.createElement('span');
    classNameTag.classList.add('column', 'is-flex', 'is-centered-flex');
    const backgroundColor = probabilityToColor(probability);
    classNameTag.style.backgroundColor = backgroundColor;

    const classNameTagContent = document.createElement('span');
    classNameTagContent.innerText = className;
    classNameTagContent.style.background = 'transparent';
    classNameTagContent.style.color = invertColor(backgroundColor, true);

    classNameTag.appendChild(classNameTagContent);
    tags.appendChild(probabilityTag);
    tags.appendChild(classNameTag);
    classificationList.appendChild(tags);
  });

  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const runPrediction = (modelName, input, initialisationStart) => {
  efficientnet[modelName].predict(input, 5).then((output) => {
    displayClassification(modelName, output);
    status(`Finished running ${modelName.toUpperCase()} in ${
      ((performance.now() - initialisationStart) / 1000).toFixed(2)} s`);
  });
};

const runEfficientNet = async (modelName) => {
  if (!efficientnet[modelName].hasLoaded()) {
    const loadingStart = performance.now();
    status(`Loading the model...`);
    await efficientnet[modelName].load();
    status(`Finished loading ${modelName.toUpperCase()} in ${
      ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
  }
  status(`Running the inference...`);
  await tf.nextFrame();
  const initialisationStart = performance.now();
  const isQuantizationDisabled =
      document.getElementById('is-quantization-disabled').checked;
  if (!(isQuantizationDisabled ^ state.quantizationBytes)) {
    state.quantizationBytes = isQuantizationDisabled ? 4 : 2;
    await initializeModels();
  }
  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }

  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(modelName, input, initialisationStart);
  } else {
    input.onload = () => {
      runPrediction(modelName, input, initialisationStart);
    };
  }
};
window.onload = initializeModels;
