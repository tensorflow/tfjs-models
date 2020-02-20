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

import * as faceMesh from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');
import Stats from 'stats.js';

let model, ctx, videoWidth, videoHeight, video, canvas, scatterGLHasInitialized = false;

const stats = new Stats();
const r = 1;

const state = {
  backend: 'webgl'
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(async backend => {
  await tf.setBackend(backend);
});

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {facingMode: 'user'},
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
}

const renderPrediction = async () => {
  stats.begin();
  const returnTensors = false;
  const flipHorizontal = false;
  const return3D = true;
  const predictions = await model.estimateFaces(video, returnTensors, flipHorizontal, return3D);
  ctx.drawImage(
    video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
    canvas.height);

  if (predictions) {
    predictions.forEach(prediction => {
      let keypoints = prediction.scaledMesh;
      if(returnTensors) {
        keypoints = prediction.scaledMesh.arraySync();
      }

      for (let i = 0; i < keypoints.length; i++) {
        const x = keypoints[i][0];
        const y = keypoints[i][1];
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    if(return3D) {
      const dataset = new ScatterGL.Dataset(
        predictions[0].scaledMesh.map(point =>
          ([-point[0], -point[1], -point[2]])));

      if (!scatterGLHasInitialized) {
        scatterGL.render(dataset);
      } else {
        scatterGL.updateDataset(dataset);
      }

      scatterGLHasInitialized = true;
    }
  }

  stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);

  const useVideoStream = true;
  if (useVideoStream) {
    await setupCamera();
    video.play();
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
  } else {
    video = document.querySelector("img");
    videoWidth = 640;
    videoHeight = 640;
  }

  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector(".canvas-wrapper");
  canvasContainer.style.width = `${videoWidth}px`;
  canvasContainer.style.height = `${videoHeight}px`;

  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  model = await faceMesh.load({
    maxFaces: 1
  });

  setupFPS();

  renderPrediction();
};

const scatterGL = new ScatterGL(
  document.querySelector("#scatter-gl-container"), {
    'rotateOnStart': false,
    'selectEnabled': false
  });

setupPage();
