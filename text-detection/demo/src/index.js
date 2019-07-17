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
// import * as tf from '@tensorflow/tfjs';

// const state = {
//   quantizationBytes: 2,
// };

// const textDetection = {};

// const toggleInvisible = (elementId, force = undefined) => {
//   const outputContainer = document.getElementById(elementId);
//   outputContainer.classList.toggle('is-invisible', force);
// }
// ;

const initializeModels = async () => {
  const loadingStart = performance.now();
  const model = await load();
  status(`Loaded in ${(performance.now() - loadingStart) / 1000}s`);
  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }

  const runPrediction = async () => {
    const predictionStart = performance.now();
    const boxes = await model.predict(input);
    status(`Finished in ${(performance.now() - predictionStart) / 1000} s.`);
    status(`The boxes are ${JSON.stringify(boxes)}`);
  };
  if (input.complete && input.naturalHeight !== 0) {
    await runPrediction();
  } else {
    input.onload = async () => {
      await runPrediction();
    };
  }

  // [1, 2, 4].forEach((quantizationBytes) => {
  //   if (textDetection[quantizationBytes]) {
  //     textDetection[quantizationBytes].dispose();
  //   }
  //   textDetection[quantizationBytes] = new TextDetection(quantizationBytes);
  //   // const runner = document.getElementById(`run-${quantizationBytes}`);
  //   // runner.onclick = async () => {
  //   //   toggleInvisible('classification-card', true);
  //   //   await tf.nextFrame();
  //   //   await runTextDetection(quantizationBytes);
  //   // };
  // });
  // const uploader = document.getElementById('upload-image');
  // uploader.addEventListener('change', processImages);
  // status('Initialised models, waiting for input...');
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

// const runPrediction = (quantizationBytes, input, initialisationStart) => {
//   textDetection[quantizationBytes].predict(input).then((output) => {
//     // displayClassification(quantizationBytes, output);
//     status(`Finished running ${quantizationBytes.toUpperCase()} in ${
//       ((performance.now() - initialisationStart) / 1000).toFixed(2)} s`);
//   });
// };

// const runTextDetection = async (modelName) => {
//   if (!textDetection[modelName].hasLoaded()) {
//     const loadingStart = performance.now();
//     status(`Loading the model...`);
//     await textDetection[modelName].load();
//     status(`Finished loading ${modelName.toUpperCase()} in ${
//       ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
//   }
//   status(`Running the inference...`);
//   await tf.nextFrame();
//   const initialisationStart = performance.now();
//   const isQuantizationDisabled =
//       document.getElementById('is-quantization-disabled').checked;
//   if (!(isQuantizationDisabled ^ state.quantizationBytes)) {
//     state.quantizationBytes = isQuantizationDisabled ? 4 : 2;
//     await initializeModels();
//   }
//   const input = document.getElementById('input-image');
//   if (!input.src || !input.src.length || input.src.length === 0) {
//     status('Failed! Please load an image first.');
//     return;
//   }

//   if (input.complete && input.naturalHeight !== 0) {
//     runPrediction(modelName, input, initialisationStart);
//   } else {
//     input.onload = () => {
//       runPrediction(modelName, input, initialisationStart);
//     };
//   }
// };

// const setImage = (src) => {
//   toggleInvisible('classification-card', true);
//   const image = document.getElementById('input-image');
//   image.src = src;
//   toggleInvisible('input-card', false);
//   status('Waiting until the model is picked...');
// };

// const processImage = (file) => {
//   if (!file.type.match('image.*')) {
//     return;
//   }
//   const reader = new FileReader();
//   reader.onload = (event) => {
//     setImage(event.target.result);
//   };
//   reader.readAsDataURL(file);
// };

// const processImages = (event) => {
//   const files = event.target.files;
//   Array.from(files).forEach(processImage);
// };

// const probabilityToColor = (probability) => {
//   const colors = [
//     '#a50026',
//     '#d73027',
//     '#f46d43',
//     '#fdae61',
//     '#fee08b',
//     '#ffffbf',
//     '#d9ef8b',
//     '#a6d96a',
//     '#66bd63',
//     '#1a9850',
//     '#006837',
//   ];
//   return colors[Math.round(probability * (colors.length - 1))];
// };

// const displayClassification = (modelName, classification) => {
//   toggleInvisible('classification-card', false);
//   const classificationList = document.getElementById('classification');
//   while (classificationList.firstChild) {
//     classificationList.removeChild(classificationList.firstChild);
//   }

//   classification.forEach(({probability, className}) => {
//     const tags = document.createElement('div');
//     tags.classList.add('column', 'is-inline-flex');

//     const probabilityTag = document.createElement('span');
//     probabilityTag.classList.add('tag', 'is-dark');
//     probabilityTag.innerText = probability.toFixed(4);
//     probabilityTag.style.height = 'auto';
//     probabilityTag.style.borderRadius = '0';

//     const classNameTag = document.createElement('span');
//     classNameTag.classList.add('column', 'is-flex', 'is-centered-flex');
//     classNameTag.style.backgroundColor = probabilityToColor(probability);

//     const classNameTagContent = document.createElement('span');
//     classNameTagContent.innerText = className;
//     classNameTagContent.style.background = 'inherit';
//     classNameTagContent.style.backgroundClip = 'text';
//     classNameTagContent.style.color = 'transparent';
//     classNameTagContent.style.filter = 'invert(1) grayscale() contrast(100)';

//     classNameTag.appendChild(classNameTagContent);
//     tags.appendChild(probabilityTag);
//     tags.appendChild(classNameTag);
//     classificationList.appendChild(tags);
//   });

//   const inputContainer = document.getElementById('input-card');
//   inputContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
// };

window.onload = initializeModels;
