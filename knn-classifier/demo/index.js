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
import * as mobilenetModule from '@tensorflow-models/mobilenet';

import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import Stats from 'stats.js';

import * as knnClassifier from '@tensorflow-models/knn-classifier';

const videoWidth = 300;
const videoHeight = 250;
const stats = new Stats();

// Number of classes to classify
const NUM_CLASSES = 3;

// K value for KNN
const TOPK = 3;

const infoTexts = [];
let training = -1;
let classifier;
let mobilenet;
let video;

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

/**
 * Setup training GUI. Adds a training button for each class,
 * and sets up mouse events.
 */
function setupGui() {
  // Create training buttons and info texts
  for (let i = 0; i < NUM_CLASSES; i++) {
    const div = document.createElement('div');
    document.body.appendChild(div);
    div.style.marginBottom = '10px';

    // Create training button
    const button = document.createElement('button');
    button.innerText = 'Train ' + i;
    div.appendChild(button);

    // Listen for mouse events when clicking the button
    button.addEventListener('click', () => {
      training = i;
    });

    // Create info text
    const infoText = document.createElement('span');
    infoText.innerText = ' No examples added';
    div.appendChild(infoText);
    infoTexts.push(infoText);
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
 * Animation function called on each frame, running prediction
 */
async function animate() {
  stats.begin();

  // Get image data from video element
  const image = tf.browser.fromPixels(video);
  let logits;
  // 'conv_preds' is the logits activation of MobileNet.
  const infer = () => mobilenet.infer(image, 'conv_preds');

  // Train class if one of the buttons is held down
  if (training != -1) {
    logits = infer();
    // Add current image to classifier
    classifier.addExample(logits, training);

    // Reset the training bit so we only collect during clicks.
    training = -1;
  }

  // If the classifier has examples for any classes, make a prediction!
  const numClasses = classifier.getNumClasses();
  if (numClasses > 0) {
    logits = infer();

    const res = await classifier.predictClass(logits, TOPK);
    for (let i = 0; i < NUM_CLASSES; i++) {
      // Make the predicted class bold
      if (res.label == `${i}`) {
        infoTexts[i].style.fontWeight = 'bold';
      } else {
        infoTexts[i].style.fontWeight = 'normal';
      }

      const classExampleCount = classifier.getClassExampleCount();
      // Update info text
      if (classExampleCount[i] > 0) {
        const conf = res.confidences[i] * 100;
        infoTexts[i].innerText = ` ${classExampleCount[i]} examples - ${conf}%`;
      }
    }
  }

  image.dispose();
  if (logits != null) {
    logits.dispose();
  }

  stats.end();

  requestAnimationFrame(animate);
}

/**
 * Kicks off the demo by loading the knn model, finding and loading
 * available camera devices, and setting off the animate function.
 */
export async function bindPage() {
  classifier = knnClassifier.create();
  mobilenet = await mobilenetModule.load();

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  // Setup the GUI
  setupGui();
  setupFPS();

  // Setup the camera
  try {
    video = await setupCamera();
    video.play();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  // Start animation loop
  animate();
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
