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
import { EfficientNet } from '@tensorflow-models/efficientnet';
import * as tf from '@tensorflow/tfjs';

const state = {
  isQuantized: true,
};

const efficientnet = {
  b0: undefined,
  b1: undefined,
  b2: undefined,
  b3: undefined,
  b4: undefined,
  b5: undefined,
};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  Object.keys(efficientnet).forEach((modelName) => {
    if (efficientnet[modelName]) {
      efficientnet[modelName].dispose();
    }
    efficientnet[modelName] = new EfficientNet(modelName, state.isQuantized);
    const runner = document.getElementById(`run-${modelName}`);
    runner.onclick = async () => {
      runner.classList.add('is-loading');
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
  return colors[
    probability < 1
      ? Math.floor(probability * colors.length)
      : colors.length - 1
  ];
};

const displayClassification = (modelName, classification) => {
  toggleInvisible('classification-card', false);
  const classificationList = document.getElementById('classification');
  while (classificationList.firstChild) {
    classificationList.removeChild(classificationList.firstChild);
  }

  console.log(classification);
  classification.forEach(({ probability, className }) => {
    const tags = document.createElement('div');
    tags.classList.add('column', 'is-inline-flex');
    const probabilityTag = document.createElement('span');
    probabilityTag.classList.add('tag', 'is-dark');
    probabilityTag.innerText = probability.toFixed(4);
    probabilityTag.style.height = 'auto';
    probabilityTag.style.borderRadius = '0';
    const classNameTag = document.createElement('span');
    classNameTag.classList.add('column', 'is-flex', 'is-centered-flex');
    classNameTag.style.backgroundColor = probabilityToColor(probability);
    const classNameTagContent = document.createElement('span');
    classNameTagContent.innerText = className;
    classNameTagContent.style.background = 'inherit';
    classNameTagContent.style.backgroundClip = 'text';
    classNameTagContent.style.color = 'transparent';
    classNameTagContent.style.filter = 'invert(1) grayscale() contrast(100)';
    // classNameTagContent.style.color = '#fff';
    // classNameTagContent.style.mixBlendMode = 'difference';
    classNameTag.appendChild(classNameTagContent);
    tags.appendChild(probabilityTag);
    tags.appendChild(classNameTag);
    classificationList.appendChild(tags);
  });

  const runner = document.getElementById(`run-${modelName}`);
  runner.classList.remove('is-loading');

  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
};

const runPrediction = (modelName, input, initialisationStart) => {
  const model = efficientnet[modelName];
  model.predict(input, 5).then((output) => {
    displayClassification(modelName, output);
    status(
        `Finished running ${modelName.toUpperCase()} in ${(
          (performance.now() - initialisationStart) /
        1000
        ).toFixed(2)} s`
    );
  });
};

const runEfficientNet = async (modelName) => {
  status(`Running the inference...`);
  await tf.nextFrame();
  const initialisationStart = performance.now();
  const isQuantizationDisabled = document.getElementById(
      'is-quantization-disabled'
  ).checked;
  if (!(isQuantizationDisabled ^ state.isQuantized)) {
    state.isQuantized = !isQuantizationDisabled;
    await initializeModels();
  }
  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    const runner = document.getElementById(`run-${modelName}`);
    runner.classList.remove('is-loading');
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
