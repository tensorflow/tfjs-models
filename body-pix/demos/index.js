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

import * as partColorScales from './part_color_scales';

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

function isSafari() {
  return (/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
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
  estimate: 'segmentation',
  input:
      {mobileNetArchitecture: isMobile() ? '0.50' : '0.75', outputStride: 16},
  segmentation: {
    segmentationThreshold: 0.5,
    effect: 'mask',
    maskBackground: true,
    opacity: 0.7,
    backgroundBlurAmount: 3,
    maskBlurAmount: 0,
    // on safari, blurring happens on the cpu, thus reducing performance, so
    // default to turning this off for safari
    edgeBlurAmount: isSafari() ? 0 : 3
  },
  partMap: {colorScale: 'warm', segmentationThreshold: 0.5},
  net: null,
};


/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // Architecture: there are a few BodyPix models varying in size and
  // accuracy. 1.00 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = gui.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.00', '0.75', '0.50', '0.25']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  gui.add(guiState.input, 'outputStride', [8, 16, 32]);

  const estimateController =
      gui.add(guiState, 'estimate', ['segmentation', 'partmap']);

  let segmentation = gui.addFolder('Segmentation');
  segmentation.add(guiState.segmentation, 'segmentationThreshold', 0.0, 1.0);
  const segmentationEffectController =
      segmentation.add(guiState.segmentation, 'effect', ['mask', 'bokeh']);

  segmentation.open();

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
  partMap.add(guiState.partMap, 'colorScale', Object.keys(partColorScales))
      .onChange(colorScale => {
        setShownPartColorScales(colorScale);
      });
  setShownPartColorScales(guiState.partMap.colorScale);

  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });

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
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to BodyPix to estimate segmentation - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function segmentBodyInRealTime(video, net) {
  const canvas = document.getElementById('output');

  async function bodySegmentationFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the BodyPix model weights for either the 0.25, 0.50, 0.75, or 1.00
      // version
      guiState.net = await bodyPix.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will
    // slow down the GPU
    const outputStride = +guiState.input.outputStride;

    const flipHorizontal = true;

    switch (guiState.estimate) {
      case 'segmentation':
        const personSegmentation =
            await guiState.net.estimatePersonSegmentation(
                video, flipHorizontal, outputStride,
                guiState.segmentation.segmentationThreshold);

        switch (guiState.segmentation.effect) {
          case 'mask':
            const mask = bodyPix.toMaskImageData(
                personSegmentation, guiState.segmentation.maskBackground);
            bodyPix.drawMask(
                canvas, video, mask, guiState.segmentation.opacity,
                guiState.segmentation.maskBlurAmount, flipHorizontal);

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
            guiState.partMap.segmentationThreshold);

        const coloredPartImageOpacity = 0.7;
        const coloredPartImageData = bodyPix.toColoredPartImageData(
            partSegmentation, partColorScales[guiState.partMap.colorScale]);

        bodyPix.drawMask(
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
 * Kicks off the demo.
 */
export async function bindPage() {
  // Load the BodyPix model weights with architecture 0.75
  const net = await bodyPix.load(+guiState.input.mobileNetArchitecture);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'inline-block';

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
  segmentBodyInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
