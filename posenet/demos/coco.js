/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import dat from 'dat.gui';
import * as posenet from '../src';
import {drawKeypoints, drawSkeleton} from './demo_util';

const images = [
  'frisbee.jpg',
  'frisbee_2.jpg',
  'backpackman.jpg',
  'boy_doughnut.jpg',
  'soccer.png',
  'yoga.jpg',
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

require('./images/frisbee.jpg')
require('./images/frisbee_2.jpg')
require('./images/backpackman.jpg')
require('./images/boy_doughnut.jpg')
require('./images/soccer.png')
require('./images/with_computer.jpg')
require('./images/snowboard.jpg')
require('./images/person_bench.jpg')
require('./images/skiing.jpg')
require('./images/fire_hydrant.jpg')
require('./images/kyte.jpg')
require('./images/looking_at_computer.jpg')
require('./images/tennis.jpg')
require('./images/tennis_standing.jpg')
require('./images/truck.jpg')
require('./images/on_bus.jpg')
require('./images/tie_with_beer.jpg')
require('./images/baseball.jpg')
require('./images/multi_skiing.jpg')
require('./images/riding_elephant.jpg')
require('./images/skate_park_venice.jpg')
require('./images/skate_park.jpg')
require('./images/tennis_in_crowd.jpg')
require('./images/two_on_bench.jpg')

function toImageData(image) {
  const [height, width] = image.shape;

  const imageData = new ImageData(width, height);
  const data = image.buffer().values;

  for (let i = 0; i < height * width; i++) {
    const j = i * 4;
    const k = i * 4;

    imageData.data[j + 0] = Math.round(255 * data[k + 0]);
    imageData.data[j + 1] = Math.round(255 * data[k + 1]);
    imageData.data[j + 2] = Math.round(255 * data[k + 2]);
    imageData.data[j + 3] = Math.max(20, Math.round(255 * data[k + 3]));
  }

  return imageData;
}

function renderToCanvas(image, canvas) {
  const [height, width] = image.shape;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);

  const imageData = toImageData(image);

  ctx.putImageData(imageData, 0, 0);
}

export function drawHeatmapImage(heatmaps) {
  const singleChannelImage = posenet.toHeatmapImage(heatmaps);
  const scaledUp = posenet.resizeBilinearGrayscale(
    singleChannelImage, [100, 100]);

  renderToCanvas(scaledUp, document.getElementById('heatmap'));
}
export function drawHeatmapAsAlpha(image, heatmaps, outputStride, canvas) {
  const singleChannelImage = posenet.toHeatmapImage(heatmaps);
  const pixels = tf.fromPixels(image);
  const alphadImage = posenet.setHeatmapAsAlphaChannel(
    pixels, outputStride, singleChannelImage);

  renderToCanvas(alphadImage, canvas);
}

function drawResults(image, outputStride, poses,
  minPartConfidence, minPoseConfidence) {
  const resultsElement = document.getElementById(`results`);
  const resultsCanvas = resultsElement.querySelector('canvas');

  resultsElement.querySelector('#outputStride').innerHTML =
        String(outputStride);

  poses.forEach((pose) => {
    if (pose.score >= minPoseConfidence) {
      console.log(pose);
      drawKeypoints(pose.keypoints,
        minPartConfidence, resultsCanvas.getContext('2d'));
      drawSkeleton(pose.keypoints,
        minPartConfidence, resultsCanvas.getContext('2d'));
    }
  });
}

async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = require(`./images/${imagePath}`);
  return promise;
}

function singlePersonCanvas() {
  return document.querySelector('#single canvas');
}

function drawSinglePoseResults(pose) {
  const canvas = singlePersonCanvas();
  console.log('got canvas');
  const ctx = canvas.getContext('2d');
  drawResults(ctx, image, guiState.outputStride, [pose],
    guiState.minPartConfidence, guiState.minPoseConfidence);
}

function drawMultiplePosesResults() {
}

async function decodeSingleAndMultiplePoses(modelOutputs) {
  console.log('decoding');
  const pose = await posenet.decodeSinglePose(
    modelOutputs.heatmapScores, modelOutputs.offsets,
    guiState.singlePoseDetection.outputStride);

  const poses = await posenet.decodeMultiplePoses(
    modelOutputs.heatmapScores, modelOutputs.offsets,
    modelOutputs.displacementFwd, modelOutputs.displacementBwd,
    guiState.multiPoseDetection.outputStride,
    guiState.multiPoseDetection.maxDetections, guiState.multiPoseDetection);

  console.log('decoded');

  drawSinglePoseResults(pose);
  drawMultiplePosesResults(poses);
}

function setStatusText(text) {
  const resultElement = document.getElementById('status');
  resultElement.innerText = text;
}

let modelOutputs = null;
let image = null;

function disposeModelOutputs() {
  if (modelOutputs) {
    modelOutputs.heatmapScores.dispose();
    modelOutputs.offsets.dispose();
    modelOutputs.displacementFwd.dispose();
    modelOutputs.displacementBwd.dispose();
  }
  if (image) {
    image.dispose();
  }
}

async function testImageAndEstimatePoses(model) {
  setStatusText('Predicting...');
  document.getElementById('results').style.display = 'none';

  disposeModelOutputs();
  image = await loadImage(guiState.image);

  console.log('predicting');
  modelOutputs = await model.predictForMultiPose(image, guiState.outputStride);
  console.log('predicted');

  await decodeSingleAndMultiplePoses();

  setStatusText('');
  document.getElementById('results').style.display = 'block';
}

let guiState;

function setupGui(model) {
  guiState = {
    outputStride: 8,
    image: images[0],
    detect: () => {
      testImageAndEstimatePoses(
        model, guiState.image, guiState);
    },
    singlePoseDetection: {
      minPartConfidence: 0.5,
      minPoseConfidence: 0.5,
    },
    multiPoseDetection: {
      minPartConfidence: 0.5,
      minPoseConfidence: 0.5,
      scoreThreshold: 0.5,
      nmsRadius: 13.0,
      maxDetections: 15,
    },
  };

  const gui = new dat.GUI();
  gui.add(guiState, 'outputStride', [32, 16, 8])
    .onChange((outputStride) => guiState.outputStride =
        Number(outputStride));
  gui.add(guiState, 'image', images);
  gui.add(guiState, 'detect');
  const singlePoseDetection = gui.addFolder('Single Pose Detection');
  singlePoseDetection.add(
    guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);
  singlePoseDetection.add(
    guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  singlePoseDetection.open();

  const multiPoseDetection = gui.addFolder('Multi Pose Detection');
  multiPoseDetection.open();
  multiPoseDetection.add(
    guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  multiPoseDetection.add(
    guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);

  multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0);
  multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
    .min(1)
    .max(20)
    .step(1);
}

export async function bindPage() {
  const model = new posenet.PoseNet();

  await model.load();

  setupGui(model);
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';
}

bindPage();
