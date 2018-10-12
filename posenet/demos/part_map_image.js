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
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import dat from 'dat.gui';

import {partColors} from './demo_util';
// clang-format off
import {
  drawKeypoints,
  drawPoint,
  drawSkeleton,
  renderImageToCanvas,
} from './demo_util';

// clang-format on

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

/**
 * Draws a pose if it passes a minimum confidence onto a canvas.
 * Only the pose's keypoints that pass a minPartConfidence are drawn.
 */
function drawResults(canvas, poses, minPartConfidence, minPoseConfidence) {
  poses.forEach((pose) => {
    if (pose.score >= minPoseConfidence) {
      if (guiState.showKeypoints) {
        drawKeypoints(
            pose.keypoints, minPartConfidence, canvas.getContext('2d'));
      }

      if (guiState.showSkeleton) {
        drawSkeleton(
            pose.keypoints, minPartConfidence, canvas.getContext('2d'));
      }
    }
  });
}

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

/**
 * Draw the results from the single-pose estimation on to a canvas
 */
async function drawSinglePoseResults(
    scaledPose, image, segmentationMask, coloredPartImage) {
  const canvas = singlePersonCanvas();
  canvas.height = guiState.resizedAndPadded.shape[0];
  canvas.width = guiState.resizedAndPadded.shape[1];
  // await tf.toPixels(scaledImage, canvas);
  renderImageToCanvas(image, [513, 513], canvas);

  await drawPartHeatmapAndSegmentation(
      canvas, segmentationMask, coloredPartImage);

  drawResults(
      canvas, [scaledPose], guiState.singlePoseDetection.minPartConfidence,
      guiState.singlePoseDetection.minPoseConfidence);

  const {partChannel, showSegments, showPartHeatmaps} = guiState.visualizeParts;
  const partChannelId = +partChannel;


  visualizeParts(
      partChannelId, showSegments, showPartHeatmaps, canvas.getContext('2d'));
}

const segmentationDarkening = 0.25;
const partMapDarkening = 0.3;
async function drawPartHeatmapAndSegmentation(
    canvas, segmentationMask, partMapImage) {
  const filteredImage = tf.tidy(() => {
    let result = tf.fromPixels(canvas);

    if (guiState.showSegments && !guiState.showPartHeatmaps) {
      const invertedMask = tf.scalar(1, 'int32').sub(segmentationMask);
      const darkeningMask = invertedMask.cast('float32')
                                .mul(tf.scalar(segmentationDarkening))
                                .add(segmentationMask.cast('float32'));

      result =
          result.cast('float32').mul(darkeningMask.expandDims(2)).cast('int32');
    }

    if (guiState.showPartHeatmaps) {
      const darkenedImage =
          result.cast('float32').mul(tf.scalar(partMapDarkening));

      result = darkenedImage
                   .add(partMapImage.cast('float32').mul(
                       tf.scalar(1 - partMapDarkening)))
                   .cast('int32');
    }

    return result;
  });

  await tf.toPixels(filteredImage, canvas);

  filteredImage.dispose();
}

function visualizeParts(partChannelId, showSegments, showPartHeatmaps, ctx) {
  const {segmentScores, partHeatmapScores} = modelOutputs;
  const outputStride = +guiState.outputStride;

  const [height, width] = segmentScores.shape;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const score = segmentScores.get(y, x, 0);

      // to save on performance, don't draw anything with a low score.
      if (score < 0.05) continue;

      // set opacity of drawn elements based on the score
      ctx.globalAlpha = score;

      if (showSegments) {
        drawPoint(ctx, y * outputStride, x * outputStride, 2, 'yellow');
      }

      if (showPartHeatmaps) {
        const partScore = partHeatmapScores.get(y, x, partChannelId);
        ctx.globalAlpha *= partScore;
        drawPoint(ctx, y * outputStride, x * outputStride, 2, 'red');
      }
    }
  }
}

const imageSize = 513;

const resizedHeight = 353;
const resizedWidth = 257;

/**
 * Converts the raw model output results into single-pose estimation results
 */
async function decodeSinglePoseAndDrawResults() {
  if (!modelOutputs) {
    return;
  }

  const [height, width] = [imageSize, imageSize];
  const {paddedBy} = guiState;

  const scaledSegmentScores = posenet.scaleAndCropToInputTensorShape(
      modelOutputs.segmentScores, [height, width],
      [resizedHeight, resizedWidth], paddedBy);

  const segmentationMask = posenet.toMask(
      scaledSegmentScores.squeeze(), guiState.segmentationThreshold);

  const scaledPartHeatmapScore = posenet.scaleAndCropToInputTensorShape(
      modelOutputs.partHeatmapScores, [height, width],
      [resizedHeight, resizedWidth], paddedBy);

  const coloredPartImage = posenet.decodeAndClipColoredPartMap(
      segmentationMask, scaledPartHeatmapScore, partColors);

  const pose = await posenet.decodeSinglePose(
      modelOutputs.heatmapScores, modelOutputs.offsets, guiState.outputStride);

  const [[padT, padB], [padL, padR]] = guiState.paddedBy;
  const scaleY = image.height / (resizedHeight - padT - padB);
  const scaleX = image.width / (resizedWidth - padL - padR);

  const poseWidthPaddingRemovedAndScaled =
      posenet.translateAndScalePose(pose, -padT, -padL, scaleY, scaleX);

  await drawSinglePoseResults(
      poseWidthPaddingRemovedAndScaled, image, segmentationMask,
      coloredPartImage);

  scaledSegmentScores.dispose();
  scaledPartHeatmapScore.dispose();
  segmentationMask.dispose();
  coloredPartImage.dispose();
}

function decodeSingleAndMultiplePoses() {
  decodeSinglePoseAndDrawResults();
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
    modelOutputs.heatmapScores.dispose();
    modelOutputs.offsets.dispose();
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
      posenet.resizeAndPadTo(input, [353, 257]);

  // renderImageToCanvas(resizedAndPadded, [353, 257], singlePersonCanvas());
  guiState.originalSize = input.shape;
  guiState.paddedBy = paddedBy;
  guiState.resizedAndPadded = resizedAndPadded;

  // Stores the raw model outputs from both single- and multi-pose results can
  // be decoded.
  // Normally you would call estimateSinglePose or estimateMultiplePoses,
  // but by calling this method we can previous the outputs of the model and
  // visualize them.
  modelOutputs = await net.predictForSinglePoseWithPartMap(
      resizedAndPadded, guiState.outputStride);

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
    singlePoseDetection: {
      minPartConfidence: 0.5,
      minPoseConfidence: 0.5,
    },
    multiPoseDetection: {
      minPartConfidence: 0.5,
      minPoseConfidence: 0.5,
      scoreThreshold: 0.5,
      nmsRadius: 20.0,
      maxDetections: 15,
    },
    showKeypoints: true,
    showSkeleton: true,
    segmentationThreshold: 0.5,
    showSegments: true,
    showPartHeatmaps: true,
    visualizeParts:
        {partChannel: 0, showSegments: false, showPartHeatmaps: false}
  };

  const gui = new dat.GUI();
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  gui.add(guiState, 'outputStride', [8, 16, 32]).onChange((outputStride) => {
    guiState.outputStride = +outputStride;
    testImageAndEstimatePoses(net);
  });
  gui.add(guiState, 'image', images)
      .onChange(() => testImageAndEstimatePoses(net));

  const singlePoseDetection = gui.addFolder('Single Pose Estimation');
  singlePoseDetection
      .add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0)
      .onChange(decodeSinglePoseAndDrawResults);
  singlePoseDetection
      .add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0)
      .onChange(decodeSinglePoseAndDrawResults);
  singlePoseDetection.open();

  gui.add(guiState, 'showKeypoints').onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'showSkeleton').onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'segmentationThreshold', 0.0, 1.0)
      .onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'showSegments').onChange(decodeSingleAndMultiplePoses);
  gui.add(guiState, 'showPartHeatmaps').onChange(decodeSingleAndMultiplePoses);
}

function drawPartColors() {
  const colorsDiv = document.getElementById('colors');

  const listHolder = document.createElement('ul');

  const listItems = posenet.partChannels.map((partChannelName, i) => {
    const listElement = document.createElement('li');
    const box = document.createElement('div');
    const color = partColors[i];
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
 * Kicks off the demo by loading the posenet model and estimating
 * poses on a default image
 */
export async function bindPage() {
  const net = await posenet.loadSegmentation(0.75);

  setupGui(net);

  drawPartColors();

  await testImageAndEstimatePoses(net);
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';
}

bindPage();
