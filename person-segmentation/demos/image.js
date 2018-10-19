/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import * as personSegmentation from '@tensorflow-models/person-segmentation';
import * as tf from '@tensorflow/tfjs';
import dat from 'dat.gui';

import * as partColorScales from './part_color_scales';
import {renderImageToCanvas,} from './util';

const images = [
  'frisbee.jpg',
  'frisbee_2.jpg',
  'backpackman.jpg',
  'boy_doughnut.jpg',
  'soccer.png',
  'with_computer.jpg',
  'snowboard.jpg',
  'person_bench.jpg',
  'skiing.jpg',
  'fire_hydrant.jpg',
  'kyte.jpg',
  'looking_at_computer.jpg',
  'tennis.jpg',
  'tennis_standing.jpg',
  'truck.jpg',
  'on_bus.jpg',
  'tie_with_beer.jpg',
  'baseball.jpg',
  'multi_skiing.jpg',
  'riding_elephant.jpg',
  'skate_park_venice.jpg',
  'skate_park.jpg',
  'tennis_in_crowd.jpg',
  'two_on_bench.jpg',
];

const imageBucket =
    'https://storage.googleapis.com/tfjs-models/assets/posenet/';

async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = `${imageBucket}${imagePath}`;
  return promise;
}

function singlePersonCanvas() {
  return document.querySelector('#single canvas');
}

const imageSize = 513;

/**
 * Draw the results from the segmentation estimation on to a canvas
 */
async function drawResults(image, segmentationMask, partSegmentation) {
  const canvas = singlePersonCanvas();
  canvas.height = guiState.resizedAndPadded.shape[0];
  canvas.width = guiState.resizedAndPadded.shape[1];
  renderImageToCanvas(image, [imageSize, imageSize], canvas);

  await drawPartHeatmapAndSegmentation(
      canvas, segmentationMask, partSegmentation);
}

async function drawPartHeatmapAndSegmentation(
    canvas, segmentationMask, partSegmentation) {
  if (guiState.showSegments && !guiState.showPartHeatmaps) {
    const segmentationMaskArray = await segmentationMask.data();

    await personSegmentation.maskAndDrawImageOnCanvas(
        canvas, image, segmentationMaskArray, 0.3, false);
  } else if (guiState.showPartHeatmaps) {
    const partMapArray = await partSegmentation.data();

    const scale = partColorScales[guiState.partColorScale];
    drawPartColors(scale);
    await personSegmentation.drawColoredPartImageOnCanvas(
        canvas, image, partMapArray, scale, 0.3, false);
  }
}


const resizedHeight = 353;
const resizedWidth = 257;

/**
 * Converts the model outputs into segmentation and part maps, and visualizes
 * them
 */
async function decodeAndDrawResults() {
  if (!modelOutputs) {
    return;
  }

  const [height, width] = [imageSize, imageSize];
  const {paddedBy} = guiState;

  const scaledSegmentScores = personSegmentation.scaleAndCropToInputTensorShape(
      modelOutputs.segmentScores, [height, width],
      [resizedHeight, resizedWidth], paddedBy);

  const segmentationMask = personSegmentation.toMask(
      scaledSegmentScores.squeeze(), guiState.segmentationThreshold);

  const scaledPartHeatmapScore =
      personSegmentation.scaleAndCropToInputTensorShape(
          modelOutputs.partHeatmapScores, [height, width],
          [resizedHeight, resizedWidth], paddedBy);

  const partSegmentation = await personSegmentation.decodePartSegmentation(
      segmentationMask, scaledPartHeatmapScore);

  await drawResults(image, segmentationMask, partSegmentation);

  scaledSegmentScores.dispose();
  scaledPartHeatmapScore.dispose();
  segmentationMask.dispose();
}


function decodeSingleAndMultiplePoses() {
  decodeAndDrawResults();
}

function setStatusText(text) {
  const resultElement = document.getElementById('status');
  resultElement.innerText = text;
}

let image = null;
let modelOutputs = null;

/**
 * Purges variables and frees up GPU memory using dispose() method
 */
function disposeModelOutputs() {
  if (modelOutputs) {
    modelOutputs.segmentScores.dispose();
    modelOutputs.partHeatmapScores.dispose();
  }
  if (guiState.resizedAndPadded) {
    guiState.resizedAndPadded.dispose();
  }
}

/**
 * Loads an image, feeds it into posenet the posenet model, and
 * calculates poses based on the model outputs
 */
async function testImageAndEstimatePoses(net) {
  setStatusText('Predicting...');
  document.getElementById('results').style.display = 'none';

  // Purge prevoius variables and free up GPU memory
  disposeModelOutputs();

  // Load an example image
  image = await loadImage(guiState.image);

  // Creates a tensor from an image
  const input = tf.fromPixels(image);

  const {resizedAndPadded, paddedBy} =
      personSegmentation.resizeAndPadTo(input, [353, 257]);

  guiState.originalSize = input.shape;
  guiState.paddedBy = paddedBy;
  guiState.resizedAndPadded = resizedAndPadded;

  // Stores the raw model outputs from person segmentation results that can
  // be decoded later.
  // Normally you would call estimatePersonSegmenation or
  // estimatePartSegmenation, but by calling this method we can preserve the
  // outputs of the model and visualize them.
  modelOutputs =
      await net.predictForPartMap(resizedAndPadded, guiState.outputStride);

  // Process the model outputs to convert into poses
  await decodeSingleAndMultiplePoses();

  setStatusText('');
  document.getElementById('results').style.display = 'block';
  input.dispose();
}

let guiState;

function setupGui(net) {
  guiState = {
    outputStride: 16,
    image: 'tennis_in_crowd.jpg',
    detectPoseButton: () => {
      testImageAndEstimatePoses(net);
    },
    segmentationThreshold: 0.5,
    showSegments: true,
    showPartHeatmaps: true,
    partColorScale: 'rainbow'
  };

  const gui = new dat.GUI();
  // Output stride:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The lower the value of the output
  // stride the higher the accuracy but slower the speed, the higher the value
  // the faster the speed but lower the accuracy.
  gui.add(guiState, 'outputStride', [8, 16, 32]).onChange((outputStride) => {
    guiState.outputStride = +outputStride;
    testImageAndEstimatePoses(net);
  });
  gui.add(guiState, 'image', images)
      .onChange(() => testImageAndEstimatePoses(net));

  gui.add(guiState, 'segmentationThreshold', 0.0, 1.0)
      .onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'showSegments').onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'showPartHeatmaps').onChange(decodeSingleAndMultiplePoses);

  gui.add(guiState, 'partColorScale', Object.keys(partColorScales))
      .onChange(decodeSingleAndMultiplePoses);
}

function drawPartColors(colorScale) {
  const colorsDiv = document.getElementById('colors');
  colorsDiv.innerHTML = '';

  const listHolder = document.createElement('ul');

  const listItems =
      personSegmentation.partChannels.map((partChannelName, i) => {
        const listElement = document.createElement('li');
        const box = document.createElement('div');
        const color = colorScale[i];
        box.setAttribute('class', 'color');
        box.setAttribute('style', `background-color: rgb(${color.join(', ')})`);


        listElement.appendChild(box);

        const text = document.createElement('div');
        text.innerText = partChannelName;

        listElement.appendChild(text);

        return listElement;
      });

  listItems.forEach((listItem) => listHolder.appendChild(listItem));

  colorsDiv.appendChild(listHolder);
}

/**
 * Kicks off the demo by loading the person segmentation model and estimating
 * poses on a default image
 */
export async function bindPage() {
  const net = await personSegmentation.load(0.75);

  setupGui(net);


  await testImageAndEstimatePoses(net);
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';
}

bindPage();
