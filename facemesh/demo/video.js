/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPath(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

let model, videoWidth, videoHeight, video,
    scatterGLHasInitialized = false, scatterGL;

const VIDEO_SIZE = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;
const stats = new Stats();
const state = {
  backend: 'webgl',
  maxFaces: 1,
  triangulateMesh: false
};

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['webgl', 'wasm', 'cpu'])
      .onChange(async backend => {
        await tf.setBackend(backend);
      });

  gui.add(state, 'maxFaces', 1, 20, 1).onChange(async val => {
    model = await facemesh.load({maxFaces: val});
  });

  gui.add(state, 'triangulateMesh');

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
          render ? 'inline-block' : 'none';
    });
  }
}

async function setupCamera() {
  video = document.getElementById('video');
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function renderPrediction() {
  stats.begin();

  const predictions = await model.estimateFaces(video);
  if (predictions.length > 0) {
    if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
      const pointsData = predictions.map(prediction => {
        let scaledMesh = prediction.scaledMesh;
        return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
      });

      let flattenedPointsData = [];
      for (let i = 0; i < pointsData.length; i++) {
        flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
      }
      const dataset = new ScatterGL.Dataset(flattenedPointsData);

      if (!scatterGLHasInitialized) {
        scatterGL.setPointColorer((i) => {
          if(i >= 468) {
            return "#FF2C35";
          }
          return "#157AB3";
        });
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

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();

  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);

  await setupCamera();
  videoWidth = 500;
  videoHeight = 500;
  video.width = videoWidth;
  video.height = videoHeight;

  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  model = await facemesh.load({maxFaces: state.maxFaces});
  renderPrediction();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
        `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;

    scatterGL = new ScatterGL(
        document.querySelector('#scatter-gl-container'),
        {'rotateOnStart': false, 'selectEnabled': false});
  }
};

main();
