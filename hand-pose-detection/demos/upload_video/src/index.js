/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import * as mpHands from '@mediapipe/hands';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as handdetection from '@tensorflow-models/hand-pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {setupStats} from './shared/stats_panel';
import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './shared/params';
import {setBackendAndEnvFlags} from './shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const statusElement = document.getElementById('status');

async function createDetector() {
  switch (STATE.model) {
    case handdetection.SupportedModels.MediaPipeHands:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands
        });
      }
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateHandsStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateHandsStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimateHands.
  beginEstimateHandsStats();

  const hands = await detector.estimateHands(
      camera.video,
      {flipHorizontal: false});

  endEstimateHandsStats();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (hands.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(hands);
  }
}

async function checkUpdate() {
  await checkGuiUpdate();

  requestAnimationFrame(checkUpdate);
};

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.video.currentSrc);
  const file = event.target.files[0];
  camera.source.src = URL.createObjectURL(file);

  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;

  statusElement.innerHTML = 'Video is loaded.';
}

async function runFrame() {
  if (video.paused) {
    // video has finished.
    camera.mediaRecorder.stop();
    camera.clearCtx();
    camera.video.style.visibility = 'visible';
    return;
  }
  await renderResult();
  rafId = requestAnimationFrame(runFrame);
}

async function run() {
  statusElement.innerHTML = 'Warming up model.';

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
        tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimateHands(
        warmUpTensor,
        {flipHorizontal: false});
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  camera.video.style.visibility = 'hidden';
  video.pause();
  video.currentTime = 0;
  video.play();
  camera.mediaRecorder.start();

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  await runFrame();
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  detector = await createDetector();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  checkUpdate();
};

app();
