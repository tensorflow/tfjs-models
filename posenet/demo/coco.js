/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import '@tensorflow/tfjs-backend-webgl';

import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

import dat from 'dat.gui';

import {isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';
// clang-format off
import {
  drawBoundingBox,
  drawKeypoints,
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
  renderImageToCanvas(image, [513, 513], canvas);
  const ctx = canvas.getContext('2d');
  poses.forEach((pose) => {
    if (pose.score >= minPoseConfidence) {
      if (guiState.showKeypoints) {
        drawKeypoints(pose.keypoints, minPartConfidence, ctx);
      }

      if (guiState.showSkeleton) {
        drawSkeleton(pose.keypoints, minPartConfidence, ctx);
      }

      if (guiState.showBoundingBox) {
        drawBoundingBox(pose.keypoints, ctx);
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

function multiPersonCanvas() {
  return document.querySelector('#multi canvas');
}

let image = null;
let predictedPoses = null;

/**
 * Draw the results from the multi-pose estimation on to a canvas
 */
function drawMultiplePosesResults() {
  const canvas = multiPersonCanvas();
  drawResults(
      canvas, predictedPoses, guiState.multiPoseDetection.minPartConfidence,
      guiState.multiPoseDetection.minPoseConfidence);
}

function setStatusText(text) {
  const resultElement = document.getElementById('status');
  resultElement.innerText = text;
}

/**
 * Purges variables and frees up GPU memory using dispose() method
 */
function disposePoses() {
  if (predictedPoses) {
    predictedPoses = null;
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
  disposePoses();

  // Load an example image
  image = await loadImage(guiState.image);

  // Creates a tensor from an image
  const input = tf.browser.fromPixels(image);

  // Estimates poses
  const poses = await net.estimatePoses(input, {
    flipHorizontal: false,
    decodingMethod: 'multi-person',
    maxDetections: guiState.multiPoseDetection.maxDetections,
    scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
    nmsRadius: guiState.multiPoseDetection.nmsRadius
  });
  predictedPoses = poses;

  // Draw poses.
  drawMultiplePosesResults();

  setStatusText('');
  document.getElementById('results').style.display = 'block';
  input.dispose();
}

/**
 * Reloads PoseNet, then loads an image, feeds it into posenet, and
 * calculates poses based on the model outputs
 */
async function reloadNetTestImageAndEstimatePoses(net) {
  if (guiState.net) {
    guiState.net.dispose();
  }
  toggleLoadingUI(true);
  guiState.net = await posenet.load({
    architecture: guiState.model.architecture,
    outputStride: guiState.model.outputStride,
    inputResolution: guiState.model.inputResolution,
    multiplier: guiState.model.multiplier,
    quantBytes: guiState.model.quantBytes,
  });
  toggleLoadingUI(false);
  testImageAndEstimatePoses(guiState.net);
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 513;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 257;

let guiState = {
  net: null,
  model: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes,
  },
  image: 'tennis_in_crowd.jpg',
  multiPoseDetection: {
    minPartConfidence: 0.1,
    minPoseConfidence: 0.2,
    nmsRadius: 20.0,
    maxDetections: 15,
  },
  showKeypoints: true,
  showSkeleton: true,
  showBoundingBox: false,
};

function setupGui(net) {
  guiState.net = net;
  const gui = new dat.GUI();

  let architectureController = null;
  guiState[tryResNetButtonName] = function() {
    architectureController.setValue('ResNet50')
  };
  gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  updateTryResNetButtonDatGuiCss();

  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  const model = gui.addFolder('Model');
  model.open();
  let inputResolutionController = null;
  function updateGuiInputResolution(inputResolutionArray) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    inputResolutionController =
        model.add(guiState.model, 'inputResolution', inputResolutionArray);
    inputResolutionController.onChange(async function(inputResolution) {
      guiState.model.inputResolution = +inputResolution;
      reloadNetTestImageAndEstimatePoses(guiState.net);
    });
  }
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;
  function updateGuiOutputStride(outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    outputStrideController =
        model.add(guiState.model, 'outputStride', outputStrideArray);
    outputStrideController.onChange((outputStride) => {
      guiState.model.outputStride = +outputStride;
      reloadNetTestImageAndEstimatePoses(guiState.net);
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;
  function updateGuiMultiplier(multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    multiplierController =
        model.add(guiState.model, 'multiplier', multiplierArray);
    multiplierController.onChange((multiplier) => {
      guiState.model.multiplier = +multiplier;
      reloadNetTestImageAndEstimatePoses(guiState.net);
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;
  function updateGuiQuantBytes(quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    quantBytesController =
        model.add(guiState.model, 'quantBytes', quantBytesArray);
    quantBytesController.onChange((quantBytes) => {
      guiState.model.quantBytes = +quantBytes;
      reloadNetTestImageAndEstimatePoses(guiState.net);
    });
  }

  function updateGui() {
    updateGuiInputResolution([257, 353, 449, 513, 801]);
    if (guiState.model.architecture.includes('ResNet50')) {
      updateGuiOutputStride([32, 16]);
      updateGuiMultiplier([1.0]);
    } else {
      updateGuiOutputStride([8, 16]);
      updateGuiMultiplier([0.5, 0.75, 1.0]);
    }
    updateGuiQuantBytes([1, 2, 4])
  }

  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController =
      model.add(guiState.model, 'architecture', ['MobileNetV1', 'ResNet50']);
  architectureController.onChange(async function(architecture) {
    if (architecture.includes('ResNet50')) {
      guiState.model.inputResolution = defaultResNetInputResolution;
      guiState.model.outputStride = defaultResNetStride;
      guiState.model.multiplier = defaultResNetMultiplier;
    } else {
      guiState.model.inputResolution = defaultMobileNetInputResolution;
      guiState.model.outputStride = defaultMobileNetStride;
      guiState.model.multiplier = defaultMobileNetMultiplier;
    }
    guiState.model.quantBytes = defaultQuantBytes;
    guiState.model.architecture = architecture;
    updateGui()
    reloadNetTestImageAndEstimatePoses(guiState.net);
  });

  updateGui();

  gui.add(guiState, 'image', images)
      .onChange(() => testImageAndEstimatePoses(guiState.net));
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
  multiPoseDetection.open();
  multiPoseDetection
      .add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0)
      .onChange(drawMultiplePosesResults);
  multiPoseDetection
      .add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0)
      .onChange(drawMultiplePosesResults);

  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0)
      .onChange(() => testImageAndEstimatePoses(guiState.net));
  multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
      .min(1)
      .max(20)
      .step(1)
      .onChange(() => testImageAndEstimatePoses(guiState.net));
  gui.add(guiState, 'showKeypoints').onChange(drawMultiplePosesResults);
  gui.add(guiState, 'showSkeleton').onChange(drawMultiplePosesResults);
  gui.add(guiState, 'showBoundingBox').onChange(drawMultiplePosesResults);
}

/**
 * Kicks off the demo by loading the posenet model and estimating
 * poses on a default image
 */
export async function bindPage() {
  toggleLoadingUI(true);
  const net = await posenet.load({
    architecture: guiState.model.architecture,
    outputStride: guiState.model.outputStride,
    inputResolution: guiState.model.inputResolution,
    multiplier: guiState.model.multiplier,
    quantBytes: guiState.model.quantBytes
  });
  toggleLoadingUI(false);
  setupGui(net);
  await testImageAndEstimatePoses(net);
}

bindPage();
