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
import 'fabric';

import {load, minTextBoxArea} from '@tensorflow-models/text-detection';
import * as tf from '@tensorflow/tfjs';

import blueBirdExample from './assets/examples/blue-bird.jpg';
import gunForHireExample from './assets/examples/gun-for-hire.jpg';
import joyYoungRogersExample from './assets/examples/joy-young-rogers.jpg';
import {findContours} from './findContours';

const state = {
  processPoints: 'minarearect',
};

const textDetection = {};
const textDetectionExamples = {
  'joy-young-rogers': joyYoungRogersExample,
  'gun-for-hire': gunForHireExample,
  'blue-bird': blueBirdExample,
};
const pointProcessors = {
  'minarearect': minTextBoxArea,
  'identity': (x) => x,
  'contours': findContours,
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
  updateInput();
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
const displayBoxes = (boxes) => {
  const input = document.getElementById('input-image');
  const height = input.height;
  const width = input.width;
  const canvas =
      new fabric.StaticCanvas('output-image', {renderOnAddRemove: false});
  canvas.setWidth(width);
  canvas.setHeight(height);
  fabric.Image.fromURL(input.src, (image) => {
    const output = image.set({left: 0, top: 0, evented: false})
        .scaleToHeight(height)
        .scaleToWidth(width);

    canvas.add(output);
    if (state.processPoints !== 'identity') {
      for (let boxIdx = 0; boxIdx < boxes.length; ++boxIdx) {
        const box = boxes[boxIdx];
        const color = getRandomColor();
        if (boxIdx === boxes.length - 1) {
          canvas.on('object:added', () => toggleInvisible('overlay', true));
        }
        canvas.add(new fabric.Polygon(
            box, {fill: color, stroke: '#000', objectCaching: false}));
      };
    } else {
      for (let boxIdx = 0; boxIdx < boxes.length; ++boxIdx) {
        const box = boxes[boxIdx];
        const color = getRandomColor();
        for (let idx = 0; idx < box.length; ++idx) {
          const {x, y} = box[idx];
          if (boxIdx === boxes.length - 1 && idx === box.length - 1) {
            canvas.on('object:added', () => toggleInvisible('overlay', true));
          }
          canvas.add(new fabric.Circle({
            left: x,
            top: y,
            radius: 3,
            strokeWidth: 1,
            stroke: color,
            fill: color,
            selectable: false,
            originX: 'center',
            originY: 'center',
            objectCaching: false,
          }));
        }
      };
    }
    canvas.renderAll();
    if (boxes.length === 0) {
      toggleInvisible('overlay', true);
    }
  });

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
  const pointProcessorName = getSelectorValue('post-processing');
  state.processPoints = pointProcessorName;
  const height = input.height;
  const model = textDetection[state.quantizationBytes];
  const img = new Image();
  img.onload = async function() {
    await tf.nextFrame();
    const originalHeight = this.height;
    const originalWidth = this.width;

    if (state.processPoints === 'contours') {
      pointProcessors[state.processPoints] = (x) =>
        findContours(x, originalHeight, originalWidth);
    }
    const factor = height / originalHeight;

    const output = await model.predict(this, {
      resizeLength: getResizeLengthSlider().value,
      minTextBoxArea: getMinTextBoxAreaSlider().value,
      minConfidence: getMinConfidenceSlider().value,
      processPoints: pointProcessors[state.processPoints],
    });
    status(`Obtained ${output.length} boxes`);
    console.log(JSON.stringify(output));
    displayBoxes(scale(factor, output));
    status(`Ran in ${howManySecondsFrom(predictionStart)} seconds`);
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
