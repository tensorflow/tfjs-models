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

import {drawKeypoints, drawSkeleton} from './demo_util';
import * as partColorScales from './part_color_scales';


const stats = new Stats();

const state = {
  video: null,
  stream: null,
  net: null,
  videoConstraints: {},
  changingCamera: false,
  changingArchitecture: false
};

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
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

function stopExistingVideoCapture() {
  if (state.video && state.video.srcObject) {
    state.video.srcObject.getTracks().forEach(track => {
      track.stop();
    })
    state.video.srcObject = null;
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

// on mobile, facing mode is the preferred way to select a camera.
// Here we use the camera label to determine if its the environment or
// user facing camera
function getFacingMode(cameraLabel) {
  if (!cameraLabel) {
    return 'user';
  }
  if (cameraLabel.toLowerCase().includes('back')) {
    return 'environment';
  } else {
    return 'user';
  }
}

async function getConstraints(cameraLabel) {
  let deviceId;
  let facingMode;

  if (cameraLabel) {
    deviceId = await getDeviceIdForLabel(cameraLabel);
    // on mobile, use the facing mode based on the camera.
    facingMode = isMobile() ? getFacingMode(cameraLabel) : null;
  };
  return {deviceId, facingMode};
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

  stopExistingVideoCapture();

  const videoConstraints = await getConstraints(cameraLabel);

  const stream = await navigator.mediaDevices.getUserMedia(
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

/**
 * Loads a video to be used in the demo
 */
async function setupVideo() {
  const videoElement = document.getElementById('video');

  var result = new Promise((resolve) => {
    videoElement.onplay = () => {
      console.log('loaded metadata');
      videoElement.width = 1080;
      videoElement.height = 720;
      resolve(videoElement);
    };
  });
  videoElement.play();
  return result;
}

async function loadImage() {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = 'two_people.jpg';
  // image.src = 'yoda_test.png';
  // image.src = 'three_people.jpg';
  // image.src = 'three_people2.jpg';
  // image.src = 'three_people_4.jpg';
  // image.src = 'four_people.jpg';
  // image.src = 'four_people2.jpg';
  // image.src = 'nine_people.jpg';
  return promise;
}

async function loadVideo(cameraLabel) {
  try {
    // state.video = await setupCamera(cameraLabel);
    // state.video = await loadImage();
    state.video = await setupVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  state.video.play();
}

const guiState = {
  algorithm: 'multi-person',
  estimate: 'partmap',
  camera: null,
  flipHorizontal: true,
  input: {
    architecture: 'ResNet50',
    outputStride: 16,
    inputResolution: 801,
    multiplier: 1.0,
    quantBytes: 4
  },
  multiPersonDecoding: {
    scoreThreshold: 0.2,
  },
  multiPersonDecoding: {
    maxDetections: 5,
    scoreThreshold: 0.2,
    nmsRadius: 20,
    numKeypointForMatching: 17,
    refineSteps: 10
  },
  segmentation: {
    segmentationThreshold: 0.7,
    effect: 'mask',
    maskBackground: true,
    opacity: 0.7,
    backgroundBlurAmount: 3,
    maskBlurAmount: 0,
    edgeBlurAmount: 3
  },
  partMap: {
    colorScale: 'rainbow',
    segmentationThreshold: 0.5,
    applyPixelation: false,
    opacity: 0.9
  },
  showFps: !isMobile()
};

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
function setupGui(cameras) {
  const gui = new dat.GUI({width: 300});

  // gui.add(guiState, 'camera', toCameraOptions(cameras))
  //     .onChange(async function(cameraLabel) {
  //       state.changingCamera = true;

  //       await loadVideo(cameraLabel);

  //       state.changingCamera = false;
  //     });

  gui.add(guiState, 'flipHorizontal');

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-person', 'multi-person']);

  // Architecture: there are a few BodyPix models varying in size and
  // accuracy.
  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  let architectureController = null;
  architectureController =
      input.add(guiState.input, 'architecture', ['ResNet50', 'MobileNetV1']);
  guiState.architecture = guiState.input.architecture;
  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    state.changingArchitecture = true;
    guiState.input.architecture = architecture;
  });

  // Output stride:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The lower the value of the output
  // stride the higher the accuracy but slower the speed, the higher the value
  // the faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  input.add(guiState.input, 'inputResolution', [513]);
  let multiplierController = null;
  multiplierController =
      input.add(guiState.input, 'multiplier', [1.0, 0.75, 0.50]);
  guiState.multiplier = guiState.input.multiplier;
  multiplierController.onChange(function(multiplier) {
    // if architecture is ResNet50, then show ResNet50 options
    console.log('onchange');
    state.changingMultiplier = true;
    guiState.input.multiplier = multiplier;
  });

  input.add(guiState.input, 'quantBytes', [4]);


  const estimateController =
      gui.add(guiState, 'estimate', ['segmentation', 'partmap']);

  let segmentation = gui.addFolder('Segmentation');
  segmentation.add(guiState.segmentation, 'segmentationThreshold', 0.0, 1.0);
  const segmentationEffectController =
      segmentation.add(guiState.segmentation, 'effect', ['mask', 'bokeh']);
  segmentation.open();

  let singlePersonDecoding = gui.addFolder('SinglePersonDecoding');
  singlePersonDecoding.add(
      guiState.multiPersonDecoding, 'scoreThreshold', 0.0, 1.0);
  singlePersonDecoding.close();

  let multiPersonDecoding = gui.addFolder('MultiPersonDecoding');
  multiPersonDecoding.add(
      guiState.multiPersonDecoding, 'maxDetections', 0, 20, 1);
  multiPersonDecoding.add(
      guiState.multiPersonDecoding, 'scoreThreshold', 0.0, 1.0);
  multiPersonDecoding.add(guiState.multiPersonDecoding, 'nmsRadius', 0, 30, 1);
  multiPersonDecoding.add(
      guiState.multiPersonDecoding, 'numKeypointForMatching', 1, 17, 1);
  multiPersonDecoding.add(
      guiState.multiPersonDecoding, 'refineSteps', 1, 10, 1);
  multiPersonDecoding.open();

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-person':
        multiPersonDecoding.close();
        singlePersonDecoding.open();
        break;
      case 'multi-person':
        singlePersonDecoding.close();
        multiPersonDecoding.open();
        break;
    }
  });

  let darknessLevel;
  let bokehBlurAmount;
  let edgeBlurAmount;
  let maskBlurAmount;
  let maskBackground;

  segmentationEffectController.onChange(function(effectType) {
    if (effectType === 'mask') {
      if (bokehBlurAmount) {
        bokehBlurAmount.remove();
      }
      if (edgeBlurAmount) {
        edgeBlurAmount.remove();
      }
      darknessLevel =
          segmentation.add(guiState.segmentation, 'opacity', 0.0, 1.0);
      maskBlurAmount = segmentation.add(guiState.segmentation, 'maskBlurAmount')
                           .min(0)
                           .max(20)
                           .step(1);
      maskBackground =
          segmentation.add(guiState.segmentation, 'maskBackground');
    } else if (effectType === 'bokeh') {
      if (darknessLevel) {
        darknessLevel.remove();
      }
      if (maskBlurAmount) {
        maskBlurAmount.remove();
      }
      if (maskBackground) {
        maskBackground.remove();
      }
      bokehBlurAmount = segmentation
                            .add(
                                guiState.segmentation,
                                'backgroundBlurAmount',
                                )
                            .min(1)
                            .max(20)
                            .step(1);
      edgeBlurAmount = segmentation.add(guiState.segmentation, 'edgeBlurAmount')
                           .min(0)
                           .max(20)
                           .step(1);
    }
  });

  // manually set the effect so that the options are shown.
  segmentationEffectController.setValue(guiState.segmentation.effect);

  let partMap = gui.addFolder('Part Map');
  partMap.add(guiState.partMap, 'segmentationThreshold', 0.0, 1.0);
  partMap.add(guiState.partMap, 'applyPixelation');
  partMap.add(guiState.partMap, 'opacity', 0.0, 1.0);
  partMap.add(guiState.partMap, 'colorScale', Object.keys(partColorScales))
      .onChange(colorScale => {
        setShownPartColorScales(colorScale);
      });
  setShownPartColorScales(guiState.partMap.colorScale);

  estimateController.onChange(function(estimationType) {
    if (estimationType === 'segmentation') {
      segmentation.open();
      partMap.close();
      document.getElementById('colors').style.display = 'none';
    } else {
      segmentation.close();
      partMap.open();
      document.getElementById('colors').style.display = 'inline-block';
    }
  });

  gui.add(guiState, 'showFps').onChange(showFps => {
    if (showFps) {
      document.body.appendChild(stats.dom);
    } else {
      document.body.removeChild(stats.dom);
    }
  })
}

function setShownPartColorScales(colorScale) {
  const colors = document.getElementById('colors');
  colors.innerHTML = '';

  const partColors = partColorScales[colorScale];
  const partNames = bodyPix.partChannels;

  for (let i = 0; i < partColors.length; i++) {
    const partColor = partColors[i];
    const child = document.createElement('li');

    child.innerHTML = `
        <div class='color' style='background-color:rgb(${partColor[0]},${
        partColor[1]},${partColor[2]})' ></div>
        ${partNames[i]}`;

    colors.appendChild(child);
  }
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  if (guiState.showFps) {
    document.body.appendChild(stats.dom);
  }
}

async function estimateSegmentation() {
  let allPersonSegmentation = null;
  console.log(guiState.algorithm);
  switch (guiState.algorithm) {
    case 'multi-person':
      allPersonSegmentation =
          await state.net.estimateMultiplePersonSegmentation(state.video, {
            segmentationThreshold: guiState.segmentation.segmentationThreshold,
            maxDetections: guiState.multiPersonDecoding.maxDetections,
            scoreThreshold: guiState.multiPersonDecoding.scoreThreshold,
            nmsRadius: guiState.multiPersonDecoding.nmsRadius,
            numKeypointForMatching:
                guiState.multiPersonDecoding.numKeypointForMatching,
            refineSteps: guiState.multiPersonDecoding.refineSteps
          });
      break;
    case 'single-person':
      const personSegmentation =
          await state.net.estimateSinglePersonSegmentation(state.video, {
            segmentationThreshold: guiState.segmentation.segmentationThreshold
          });
      allPersonSegmentation = [personSegmentation];
      break;
    default:
      break;
  };
  return allPersonSegmentation;
}

async function estimatePartSegmentation() {
  let allPersonPartSegmentation = null;
  switch (guiState.algorithm) {
    case 'multi-person':
      allPersonPartSegmentation =
          await state.net.estimateMultiplePersonPartSegmentation(state.video, {
            segmentationThreshold: guiState.segmentation.segmentationThreshold,
            maxDetections: guiState.multiPersonDecoding.maxDetections,
            scoreThreshold: guiState.multiPersonDecoding.scoreThreshold,
            nmsRadius: guiState.multiPersonDecoding.nmsRadius,
            numKeypointForMatching:
                guiState.multiPersonDecoding.numKeypointForMatching,
            refineSteps: guiState.multiPersonDecoding.refineSteps
          });
      break;
    case 'single-person':
      const personPartSegmentation =
          await state.net.estimateSinglePersonPartSegmentation(state.video, {
            segmentationThreshold: guiState.segmentation.segmentationThreshold
          });
      allPersonPartSegmentation = [personPartSegmentation];
      break;
    default:
      break;
  };
  return allPersonPartSegmentation;
}

/**
 * Feeds an image to BodyPix to estimate segmentation - this is where the
 * magic happens. This function loops with a requestAnimationFrame method.
 */
function segmentBodyInRealTime() {
  const canvas = document.getElementById('output');
  // since images are being fed from a webcam

  async function bodySegmentationFrame() {
    // if changing the model or the camera, wait a second for it to complete
    // then try again.
    if (state.changingArchitecture || state.changingMultiplier ||
        state.changingCamera) {
      // setTimeout(bodySegmentationFrame, 1000);
      console.log('changing architecture');
      console.log(guiState.input.architecture);
      state.net = await bodyPix.load({
        architecture: guiState.input.architecture,
        outputStride: guiState.input.outputStride,
        inputResolution: guiState.input.inputResolution,
        multiplier: guiState.input.multiplier,
        quantBytes: guiState.input.quantBytes
      });
      console.log('loaded');
      state.changingArchitecture = false;
      state.changingMultiplier = false;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will
    // slow down the GPU
    const outputStride = +guiState.input.outputStride;

    const flipHorizontally = guiState.flipHorizontal;

    switch (guiState.estimate) {
      case 'segmentation':
        const allPersonSegmentation = await estimateSegmentation();
        switch (guiState.segmentation.effect) {
          case 'mask':
            const ctx = canvas.getContext('2d');
            const foregroundColor = {r: 255, g: 255, b: 255, a: 255};
            const backgroundColor = {r: 0, g: 0, b: 0, a: 255};
            const mask = bodyPix.toMultiPersonMaskImageData(
                allPersonSegmentation, foregroundColor, backgroundColor, true);

            bodyPix.drawMask(
                canvas, state.video, mask, guiState.segmentation.opacity,
                guiState.segmentation.maskBlurAmount, flipHorizontally);

            allPersonSegmentation.forEach(personSegmentation => {
              let pose = personSegmentation.pose;
              if (flipHorizontally) {
                pose = bodyPix.flipPoseHorizontal(pose, mask.width);
              }
              drawKeypoints(pose.keypoints, 0.1, ctx);
              drawSkeleton(pose.keypoints, 0.1, ctx);
            });
            break;
          case 'bokeh':
            bodyPix.drawMultiPersonBokehEffect(
                canvas, state.video, allPersonSegmentation,
                +guiState.segmentation.backgroundBlurAmount,
                guiState.segmentation.edgeBlurAmount, flipHorizontally);
            break;
        }

        break;
      case 'partmap':
        const ctx = canvas.getContext('2d');
        const allPersonPartSegmentation = await estimatePartSegmentation();
        const coloredPartImageData = bodyPix.toMultiPersonColoredPartImageData(
            allPersonPartSegmentation,
            partColorScales[guiState.partMap.colorScale]);

        const maskBlurAmount = 0;
        if (guiState.partMap.applyPixelation) {
          const pixelCellWidth = 10.0;

          bodyPix.drawPixelatedMask(
              canvas, state.video, coloredPartImageData,
              guiState.partMap.opacity, maskBlurAmount, flipHorizontally,
              pixelCellWidth);
        } else {
          bodyPix.drawMask(
              canvas, state.video, coloredPartImageData, guiState.opacity,
              maskBlurAmount, flipHorizontally);
        }

        allPersonPartSegmentation.forEach(personPartSegmentation => {
          let pose = personPartSegmentation.pose;
          if (flipHorizontally) {
            pose =
                bodyPix.flipPoseHorizontal(pose, personPartSegmentation.width);
          }
          drawKeypoints(pose.keypoints, 0.1, ctx);
          drawSkeleton(pose.keypoints, 0.1, ctx);
        });

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
 * Kicks off the demo.
 */
export async function bindPage() {
  // Load the BodyPix model weights with architecture 0.75
  state.net = await bodyPix.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes
  });

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'inline-block';

  await loadVideo();

  let cameras = await getVideoInputs();

  setupFPS();
  setupGui(cameras);

  segmentBodyInRealTime();
}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
