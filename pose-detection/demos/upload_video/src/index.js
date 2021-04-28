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

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setBackendAndEnvFlags} from './util';

let detector, camera;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.MediapipeBlazepose:
      return posedetection.createDetector(STATE.model, {quantBytes: 4});
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, {modelType});
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged) {
    detector.dispose();
    detector = await createDetector(STATE.model);
    STATE.isModelChanged = false;
  }

  if (STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;
    detector.dispose();
    await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

async function renderResult(simulatedTimestamp) {
  if (video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  const poses = await detector.estimatePoses(
      camera.video,
      {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false},
      simulatedTimestamp);

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }
}

async function checkUpdate() {
  await checkGuiUpdate();

  requestAnimationFrame(checkUpdate);
};

async function app() {
  await tf.setBackend(STATE.backend);

  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  detector = await createDetector();

  camera = new Context();

  camera.video.onseeked = async () => {
    await renderResult();
  };

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  checkUpdate();
};

async function updateVideo(event) {
  URL.revokeObjectURL(camera.video.currentSrc);
  const file = event.target.files[0];
  camera.source.src = URL.createObjectURL(file);
  camera.video.load();

  await new Promise((resolve) => {
    camera.video.onloadedmetadata = () => {
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
}

function run() {
  camera.video.currentTime = 0.0;
  // Advance to next frame.
  const actualInterval = 1 / 30;
  const actualIntervalInMilliseconds = actualInterval * 1000;
  const timer = setInterval(() => {
    const nextTime = camera.video.currentTime + actualInterval;

    if (nextTime < camera.video.duration) {
      video.currentTime = nextTime;
    } else {
      camera.clearCtx();
      clearInterval(timer);
    }
  }, actualIntervalInMilliseconds);
}

app();
