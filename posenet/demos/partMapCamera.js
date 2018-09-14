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
import * as tf from '@tensorflow/tfjs';
import dat from 'dat.gui';
import Stats from 'stats.js';

import * as posenet from '../src';

import {drawKeypoints, drawSkeleton, partColors} from './demo_util';

const stats = new Stats();
const videoWidth = 640;
const videoHeight = 480;

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'single-pose',
  input:
      {mobileNetArchitecture: isMobile() ? '0.50' : '0.75', outputStride: 16},
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
    segmentationThreshold: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showSegments: true,
    showPartHeatmaps: true,
  },
  net: null,
};


/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);
  document.body.appendChild(stats.dom);
}

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  // const algorithmController =
  //     gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  // const architectureController = input.add(
  //     guiState.input, 'mobileNetArchitecture',
  //     ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);

  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'segmentationThreshold', 0.0, 1.0);
  single.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showSegments');
  output.add(guiState.output, 'showPartHeatmaps');
  output.open();


  // architectureController.onChange(function(architecture) {
  //   guiState.changeToArchitecture = architecture;
  // });
}

const segmentationDarkening = 0.25;
const partMapDarkening = 0.3;
async function drawPartHeatmapAndSegmentation(
    canvas, video, segmentation, partMap) {
  const filteredImage = tf.tidy(() => {
    let result = video;

    if (guiState.output.showSegments && !guiState.output.showPartHeatmaps) {
      const invertedMask = tf.scalar(1, 'int32').sub(segmentation);
      const darkeningMask = invertedMask.cast('float32')
                                .mul(tf.scalar(segmentationDarkening))
                                .add(segmentation.cast('float32'));

      result =
          result.cast('float32').mul(darkeningMask.expandDims(2)).cast('int32');
    }

    if (guiState.output.showPartHeatmaps) {
      const darkenedImage =
          result.cast('float32').mul(tf.scalar(partMapDarkening));

      result =
          darkenedImage
              .add(partMap.cast('float32').mul(tf.scalar(1 - partMapDarkening)))
              .cast('int32');
    }

    return result;
  });

  await tf.toPixels(filteredImage, canvas);

  filteredImage.dispose();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture, true);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const outputStride = +guiState.input.outputStride;

    const {pose, segmentationMask, coloredPartImage} =
        await guiState.net.estimateSinglePoseWithSegmentation(
            video, flipHorizontal, outputStride,
            guiState.singlePoseDetection.segmentationThreshold, partColors);

    const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
    const minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

    const videoTensor = tf.fromPixels(video).reverse(1);

    await drawPartHeatmapAndSegmentation(
        canvas, videoTensor, segmentationMask, coloredPartImage);
    videoTensor.dispose();
    if (segmentationMask) {
      segmentationMask.dispose();
    }
    if (coloredPartImage) {
      coloredPartImage.dispose();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    const {score, keypoints} = pose;
    if (score >= minPoseConfidence) {
      if (guiState.output.showPoints) {
        drawKeypoints(keypoints, minPartConfidence, ctx);
      }
      if (guiState.output.showSkeleton) {
        drawSkeleton(keypoints, minPartConfidence, ctx);
      }
    }

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75, true);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupFPS();
  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
