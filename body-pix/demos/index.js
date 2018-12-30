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
import * as bodyPix from '@tensorflow-models/body-pix';
import dat from 'dat.gui';
import Stats from 'stats.js';
import {getLeadingCommentRanges} from 'typescript';

import * as partColorScales from './part_color_scales';

const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

function isSafari() {
  return (/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
}

async function getVideoInputs() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log('enumerateDevices() not supported.');
    return [];
  }

  const devices = await navigator.mediaDevices.enumerateDevices();

  const videoDevices = devices.filter(device => device.kind === 'videoinput');

  return videoDevices;
}

let stream;

function stopExistingStream() {
  if (stream) {
    stream.getTracks().forEach(track => {
      track.stop();
    })
  }
}

async function getDeviceIdForLabel(cameraLabel) {
  const videoInputs = await getVideoInputs();

  for (let i = 0; i < videoInputs.length; i++) {
    const videoInput = videoInputs[i];
    if (videoInput.label === cameraLabel) {
      return videoInput.deviceId;
    }
  }

  return null;
}

function getFacingMode(cameraLabel) {
  if (!cameraLabel) {
    return null;
  }
  if (cameraLabel.toLowerCase().includes('back')) {
    return 'environment';
  } else {
    return 'user';
  }
}


async function getConstraints(cameraLabel) {
  if (!cameraLabel) {
    return {}
  };
  if (isiOS()) {
    return {facingMode: getFacingMode(cameraLabel)};
  } else {
    const deviceId = await getDeviceIdForLabel(cameraLabel);
    return {
      deviceId
    }
  }
}


/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera(cameraLabel) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const videoElement = document.getElementById('video');

  stopExistingStream();

  const videoConstraints = await getConstraints(cameraLabel);

  stream = await navigator.mediaDevices.getUserMedia(
      {'audio': false, 'video': videoConstraints});
  videoElement.srcObject = stream;

  return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      videoElement.width = videoElement.videoWidth;
      videoElement.height = videoElement.videoHeight;
      resolve(videoElement);
    };
  });
}


let video;

async function loadVideo(cameraLabel) {
  try {
    video = await setupCamera(cameraLabel);
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  video.play();
}

const guiState = {
  estimate: 'segmentation',
  input:
      {mobileNetArchitecture: isMobile() ? '0.50' : '0.75', outputStride: 16},
  segmentation: {
    segmentationThreshold: 0.5,
    effect: 'mask',
    opacity: 0.7,
    backgroundBlurAmount: 3,
    // on safari, blurring happens on the cpu, thus reducing performance, so
    // default to turning this off for safari
    edgeBlurAmount: isSafari() ? 0 : 3
  },
  partMap: {colorScale: 'warm'},
  net: null,
  camera: null,
  flipHorizontal: true
};


/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);
  document.body.appendChild(stats.dom);
}

function toCameraOptions(cameras) {
  const result = {default: null};

  cameras.forEach(camera => {
    result[camera.label] = camera.label;
  })

  return result;
}

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  console.log(stream, video, cameras);
  const gui = new dat.GUI({width: 300});

  gui.add(guiState, 'camera', toCameraOptions(cameras))
      .onChange(async function(cameraLabel) {
        guiState.changingCamera = true;

        if (cameraLabel && cameraLabel.toLowerCase().includes('back')) {
          guiState.flipHorizontal = false;
        }

        await loadVideo(cameraLabel);

        guiState.changingCamera = false;
      });

  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.00 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = gui.add(
                                        guiState.input, 'mobileNetArchitecture',
                                        ['1.00', '0.75', '0.50', '0.25'])
                                     .onChange(async function(architecture) {
                                       guiState.changingArchitecture = true;
                                       // Important to purge variables and free
                                       // up GPU memory
                                       guiState.net.dispose();

                                       // Load the PoseNet model weights for
                                       // either the 0.50, 0.75, 1.00, or 1.01
                                       // version
                                       guiState.net =
                                           await bodyPix.load(+architecture);

                                       guiState.changingArchitecture = false;
                                     });
  ;
  // Output stride:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The lower the value of the output
  // stride the higher the accuracy but slower the speed, the higher the value
  // the faster the speed but lower the accuracy.
  gui.add(guiState.input, 'outputStride', [8, 16, 32]);

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose
  // works for more than 1 person
  const estimateController =
      gui.add(guiState, 'estimate', ['segmentation', 'partmap']);


  let segmentation = gui.addFolder('Segmentation');
  segmentation.add(guiState.segmentation, 'segmentationThreshold', 0.0, 1.0);
  const segmentationEffectController =
      segmentation.add(guiState.segmentation, 'effect', ['mask', 'bokeh']);
  segmentation.add(guiState.segmentation, 'edgeBlurAmount')
      .min(0)
      .max(20)
      .step(1);
  segmentation.open();

  let darknessLevel;
  let bokehBlurAmount;

  segmentationEffectController.onChange(function(effectType) {
    if (effectType === 'mask') {
      if (bokehBlurAmount) {
        bokehBlurAmount.remove();
      }
      darknessLevel =
          segmentation.add(guiState.segmentation, 'opacity', 0.0, 1.0);
    } else if (effectType === 'bokeh') {
      if (darknessLevel) {
        darknessLevel.remove();
      }
      bokehBlurAmount = segmentation
                            .add(
                                guiState.segmentation,
                                'backgroundBlurAmount',
                                )
                            .min(1)
                            .max(20)
                            .step(1);
    }
  });

  // set the mask value in the segmentation effect so that the options are
  // shown.
  segmentationEffectController.setValue(guiState.segmentation.effect);

  let partMap = gui.addFolder('Part Map');
  partMap.add(guiState.partMap, 'colorScale', Object.keys(partColorScales));

  estimateController.onChange(function(estimationType) {
    if (estimationType === 'segmentation') {
      segmentation.open();
      partMap.close();
    } else {
      segmentation.close();
      partMap.open();
    }
  });
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
function segmentBodyInRealTime() {
  const canvas = document.getElementById('output');
  // since images are being fed from a webcam

  async function bodySegmentationFrame() {
    // if changing the model or the camera, wait a second and try again.
    if (guiState.changingArchitecture || guiState.changingCamera) {
      setTimeout(bodySegmentationFrame, 1000);
      return;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will
    // slow down the GPU
    const outputStride = +guiState.input.outputStride;

    // canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    const flipHorizontal = guiState.flipHorizontal;

    switch (guiState.estimate) {
      case 'segmentation':
        const personSegmentation =
            await guiState.net.estimatePersonSegmentation(
                video, flipHorizontal, outputStride,
                guiState.segmentation.segmentationThreshold);

        switch (guiState.segmentation.effect) {
          case 'mask':
            const invert = true;
            const backgroundDarkeningMask =
                bodyPix.toMaskImageData(personSegmentation, invert);
            bodyPix.drawImageWithMask(
                canvas, video, backgroundDarkeningMask,
                guiState.segmentation.opacity,
                guiState.segmentation.edgeBlurAmount, flipHorizontal);

            break;
          case 'bokeh':
            bodyPix.drawBokehEffect(
                canvas, video, personSegmentation,
                +guiState.segmentation.backgroundBlurAmount,
                guiState.segmentation.edgeBlurAmount, flipHorizontal);
            break;
        }
        break;
      case 'partmap':
        const partSegmentation = await guiState.net.estimatePartSegmentation(
            video, flipHorizontal, outputStride,
            guiState.segmentation.segmentationThreshold);

        const coloredPartImageOpacity = 0.7;
        const coloredPartImageData = bodyPix.toColoredPartImageData(
            partSegmentation, partColorScales[guiState.partMap.colorScale]);

        bodyPix.drawImageWithMask(
            canvas, video, coloredPartImageData, coloredPartImageOpacity, 0,
            flipHorizontal);

        break;
      default:
        break;
    }

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(bodySegmentationFrame);
  }

  bodySegmentationFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime
 * function.
 */
export async function bindPage() {
  // Load the BodyPix model weights with architecture 0.75
  const net = await bodyPix.load(+guiState.input.mobileNetArchitecture);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  await loadVideo();

  let cameras = await getVideoInputs();

  setupFPS();
  setupGui(cameras, net);

  segmentBodyInRealTime();
}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
