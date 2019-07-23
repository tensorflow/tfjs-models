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
// eslint-disable-next-line max-len
import 'fabric';

import {load} from '@tensorflow-models/text-detection';
import * as tf from '@tensorflow/tfjs';

import wordDemonstrationExample from './assets/examples/demonstration.jpg';
import drNoExample from './assets/examples/dr-no.jpg';
import gunForHireExample from './assets/examples/gun-for-hire.jpg';

const state = {};
const textDetection = {};
const textDetectionExamples = {
  'dr-no': drNoExample,
  'gun-for-hire': gunForHireExample,
  'demonstration': wordDemonstrationExample,
};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const howManySecondsFrom = (start) => {
  return ((performance.now() - start) / 1000).toFixed(2);
};

const getSelectorValue = (selectorId) => {
  const selector = document.getElementById(selectorId);
  return selector.options[selector.selectedIndex].value;
};

const updateSlider = (slider, outputId) => {
  document.getElementById(outputId).innerHTML = slider.value;
};

const updateInput = () => {
  toggleInvisible('output-card', true);
  const input = document.getElementById('input-image');
  const value = getSelectorValue('example');
  input.src = textDetectionExamples[value];
};

const getResizeLengthSlider = () =>
  document.getElementById('resize-length-slider');
const getMinTextBoxAreaSlider = () =>
  document.getElementById('min-textbox-area-slider');
const getMinConfidenceSlider = () =>
  document.getElementById('min-confidence-slider');

const initializeModel = async () => {
  toggleInvisible('overlay', false);
  const resizeLengthSlider = getResizeLengthSlider();
  updateSlider(resizeLengthSlider, 'resize-length-value');
  resizeLengthSlider.oninput = () =>
    updateSlider(resizeLengthSlider, 'resize-length-value');
  const minTextBoxAreaSlider = getMinTextBoxAreaSlider();
  updateSlider(minTextBoxAreaSlider, 'min-textbox-area-value');
  minTextBoxAreaSlider.oninput = () =>
    updateSlider(minTextBoxAreaSlider, 'min-textbox-area-value');
  const minConfidenceSlider = getMinConfidenceSlider();
  updateSlider(minConfidenceSlider, 'min-confidence-value');
  minConfidenceSlider.oninput = () =>
    updateSlider(minConfidenceSlider, 'min-confidence-value');
  const inputSelector = document.getElementById('example');
  inputSelector.onchange = updateInput;
  status('Loading the model...');
  const loadingStart = performance.now();
  const quantizationBytes = Number(getSelectorValue('quantizationBytes'));
  state.quantizationBytes = quantizationBytes;
  await tf.nextFrame();
  textDetection[quantizationBytes] = await load({quantizationBytes});
  const runner = document.getElementById('run');
  runner.onclick = async () => {
    toggleInvisible('output-card', true);
    await tf.nextFrame();
    await runTextDetection();
  };
  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);
  status(`Initialised the model in ${
    howManySecondsFrom(loadingStart)} seconds, waiting for input...`);
  toggleInvisible('overlay', true);
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

function getRandomColor(alpha = 0.5) {
  const num = Math.round(0xffffff * Math.random());
  const r = num >> 16;
  const g = num >> 8 & 255;
  const b = num & 255;
  return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
}
const displayBoxes = (textDetectionOutput) => {
  const input = document.getElementById('input-image');
  const height = input.height;
  const width = input.width;
  const canvas = new fabric.StaticCanvas('output-image');
  canvas.setWidth(width);
  canvas.setHeight(height);
  fabric.Image.fromURL(input.src, (image) => {
    const output = image.set({left: 0, top: 0, evented: false})
        .scaleToHeight(height)
        .scaleToWidth(width);

    canvas.add(output);
    for (const box of textDetectionOutput) {
      const color = getRandomColor();
      canvas.add(new fabric.Polygon(box, {fill: color, stroke: '#000'}));
      // for (let idx = 0; idx < box.length; ++idx) {
      //   const from = box[idx];
      //   const to = box[(idx + 1) % 4];
      //   canvas.add(new fabric.Line(
      //       [from.x, from.y, to.x, to.y],
      //       {stroke: color, strokeWidth: 2, evented: false}));
      //   // const {x, y} = box[idx];
      //   // canvas.add(new fabric.Circle({
      //   //   left: x,
      //   //   top: y,
      //   //   radius: 3,
      //   //   strokeWidth: 1,
      //   //   stroke: color,
      //   //   fill: color,
      //   //   selectable: false,
      //   //   originX: 'center',
      //   //   originY: 'center',
      //   // }));
      // }
    };
  });
  // .onload = function() {
  //   // apply the downsampling trick from
  //   // https://stackoverflow.com/a/17862644/3581829
  //   const offscreenCanvas = document.createElement('canvas');
  //   offscreenCanvas.width = this.width;
  //   offscreenCanvas.height = this.height;
  //   const offscreenCanvasContext = offscreenCanvas.getContext('2d');
  //   const steps = (offscreenCanvas.width / canvas.width) >> 1;
  //   offscreenCanvasContext.filter = `blur(${steps}px)`;
  //   offscreenCanvasContext.drawImage(this, 0, 0);
  //   ctx.drawImage(
  //       offscreenCanvas, 0, 0, offscreenCanvas.width, offscreenCanvas.height,
  //       0, 0, width, height);
  //   const boxCanvas = document.createElement('canvas');
  //   boxCanvas.width = this.width;
  //   boxCanvas.height = this.height;
  //   const boxCanvasContext = boxCanvas.getContext('2d');
  //   boxCanvasContext.strokeStyle = '#32CD32';
  //   boxCanvasContext.lineWidth = 10;
  //   for (const box of textDetectionOutput) {
  //     boxCanvasContext.beginPath();
  //     for (let idx = 0; idx < 4; ++idx) {
  //       const from = box[idx];
  //       boxCanvasContext.moveTo(from.x, from.y);
  //       const to = box[(idx + 1) % 4];
  //       boxCanvasContext.lineTo(to.x, to.y);
  //     }
  //     boxCanvasContext.stroke();
  //   };
  //   ctx.drawImage(
  //       boxCanvas, 0, 0, boxCanvas.width, boxCanvas.height, 0, 0, width,
  //       height);
  // };
  // img.src = input.src;

  toggleInvisible('output-card', false);

  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const scale = (factor, boxes) => {
  for (let boxIdx = 0; boxIdx < boxes.length; ++boxIdx) {
    const box = [];
    for (const point of boxes[boxIdx]) {
      const {x, y} = point;
      box.push({x: factor * x, y: factor * y});
    }
    boxes[boxIdx] = box;
  }
  return boxes;
};

const runPrediction = async (input, predictionStart) => {
  toggleInvisible('overlay', false);
  const height = input.height;
  const model = textDetection[state.quantizationBytes];
  const img = new Image();
  img.onload = async function() {
    await tf.nextFrame();
    const originalHeight = this.height;
    const factor = height / originalHeight;

    const output = await model.predict(this, {
      resizeLength: getResizeLengthSlider().value,
      minTextBoxArea: getMinTextBoxAreaSlider().value,
      minConfidence: getMinConfidenceSlider().value,
    });
    status(`Obtained ${output.length} boxes`);
    console.log(JSON.stringify(output));
    displayBoxes(scale(factor, output));
    status(`Ran in ${howManySecondsFrom(predictionStart)} seconds`);
    toggleInvisible('overlay', true);
  };
  img.src = input.src;
};


const runTextDetection = async () => {
  status(`Running the inference...`);
  const quantizationBytes = Number(getSelectorValue('quantizationBytes'));
  if (state.quantizationBytes !== quantizationBytes) {
    if (textDetection[quantizationBytes]) {
      textDetection[quantizationBytes].dispose();
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
    toggleInvisible('overlay', false);
    const loadingStart = performance.now();
    await tf.nextFrame();
    textDetection[quantizationBytes] = await load({quantizationBytes});
    status(`Loaded the model in ${howManySecondsFrom(loadingStart)}
    seconds`);
    toggleInvisible('overlay', true);
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

window.onload = initializeModel;
