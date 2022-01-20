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

import * as mpPose from '@mediapipe/pose';
import * as mpSelfieSegmentation from '@mediapipe/selfie_segmentation';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as poseDetection from '@tensorflow-models/pose-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let segmenter, camera, statsModel, statsE2E;
let cameras;
let modelTime = {
  startInferenceTime: 0,
  numInferences: 0,
  inferenceTimeSum: 0,
  lastPanelUpdate: 0
};
let E2ETime = {
  startInferenceTime: 0,
  numInferences: 0,
  inferenceTimeSum: 0,
  lastPanelUpdate: 0
};
let rafId;
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

async function createSegmenter() {
  switch (STATE.model) {
    case poseDetection.SupportedModels.BlazePose: {
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return poseDetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath:
              `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`,
          enableSegmentation: true,
          smoothSegmentation: true
        });
      } else if (runtime === 'tfjs') {
        return poseDetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          enableSegmentation: true,
          smoothSegmentation: true
        });
      }
    }
    case bodySegmentation.SupportedModels.BodyPix: {
      return bodySegmentation.createSegmenter(STATE.model, {
        architecture: STATE.modelConfig.architecture,
        outputStride: parseFloat(STATE.modelConfig.outputStride),
        multiplier: parseFloat(STATE.modelConfig.multiplier),
        quantBytes: parseFloat(STATE.modelConfig.quantBytes)
      });
    }
    case bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation: {
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return bodySegmentation.createSegmenter(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath:
              `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@${
                  mpSelfieSegmentation.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return bodySegmentation.createSegmenter(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
        });
      }
    }
  }
}

async function checkGuiUpdate() {
  if (STATE.isCameraChanged) {
    camera = await Camera.setupCamera(STATE.camera, cameras);
    canvas.width = camera.canvas.width;
    canvas.height = camera.canvas.height;
    STATE.isCameraChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged ||
      STATE.isVisChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (segmenter != null) {
      segmenter.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      segmenter = await createSegmenter();
    } catch (error) {
      segmenter = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
    STATE.isVisChanged = false;
  }
}

function beginEstimateSegmentationStats(time) {
  time.startInferenceTime = (performance || Date).now();
}

function endEstimateSegmentationStats(time, stats) {
  const endInferenceTime = (performance || Date).now();
  time.inferenceTimeSum += endInferenceTime - time.startInferenceTime;
  ++time.numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - time.lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = time.inferenceTimeSum / time.numInferences;
    time.inferenceTimeSum = 0;
    time.numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    time.lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let segmentation = null;

  // Segmenter can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (segmenter != null) {
    // Model FPS only counts the time it takes to finish segmentPeople.
    beginEstimateSegmentationStats(modelTime);
    // E2E FPS includes rendering time.
    beginEstimateSegmentationStats(E2ETime);

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      if (segmenter.segmentPeople != null) {
        segmentation = await segmenter.segmentPeople(camera.video, {
          flipHorizontal: false,
          multiSegmentation: false,
          segmentBodyParts: true,
          segmentationThreshold: STATE.visualization.foregroundThreshold
        });
      } else {
        segmentation = await segmenter.estimatePoses(
            camera.video, {flipHorizontal: false});
        segmentation = segmentation.map(
            singleSegmentation => singleSegmentation.segmentation);
      }
    } catch (error) {
      segmenter.dispose();
      segmenter = null;
      alert(error);
    }

    // Ensure GPU is done for timing purposes.
    const [backend] = STATE.backend.split('-');
    if (backend === 'tfjs') {
      for (const value of segmentation) {
        const mask = value.mask;
        const tensor = await mask.toTensor();

        const res = tensor.dataToGPU();

        const webGLBackend = tf.backend();
        const buffer =
            webGLBackend.gpgpu.createBufferFromTexture(res.texture, 1, 1);
        webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 1);

        res.tensorRef.dispose();
      }
    } else if (backend === 'mediapipe') {
      // Code in
      // node_modules/@mediapipe/selfie_segmentation/selfie_segmentation.js must
      // be modified to expose the webgl context it uses.
      const gl = window.exposedContext;
      if (gl)
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    }

    endEstimateSegmentationStats(modelTime, statsModel);
  }

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (segmentation && segmentation.length > 0 && !STATE.isModelChanged) {
    const vis = STATE.modelConfig.visualization;
    const options = STATE.visualization;
    if (vis === 'binaryMask') {
      const data = await bodySegmentation.toBinaryMask(
          segmentation, {r: 0, g: 0, b: 0, a: 0}, {r: 0, g: 0, b: 0, a: 255},
          false, options.foregroundThreshold);
      await bodySegmentation.drawMask(
          canvas, camera.video, data, options.maskOpacity, options.maskBlur);
    } else if (vis === 'coloredMask') {
      const data = await bodySegmentation.toColoredMask(
          segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor,
          {r: 0, g: 0, b: 0, a: 255}, options.foregroundThreshold);
      await bodySegmentation.drawMask(
          canvas, camera.video, data, options.maskOpacity, options.maskBlur);
    } else if (vis === 'pixelatedMask') {
      const data = await bodySegmentation.toColoredMask(
          segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor,
          {r: 0, g: 0, b: 0, a: 255}, options.foregroundThreshold);
      await bodySegmentation.drawPixelatedMask(
          canvas, camera.video, data, options.maskOpacity, options.maskBlur,
          false, options.pixelCellWidth);
    } else if (vis === 'bokehEffect') {
      await bodySegmentation.drawBokehEffect(
          canvas, camera.video, segmentation, options.foregroundThreshold,
          options.backgroundBlur, options.edgeBlur);
    } else if (vis === 'blurFace') {
      await bodySegmentation.blurBodyPart(
          canvas, camera.video, segmentation, [0, 1],
          options.foregroundThreshold, options.backgroundBlur,
          options.edgeBlur);
    } else {
      camera.drawFromVideo(ctx);
    }
  }
  camera.drawToCanvas(canvas);
  endEstimateSegmentationStats(E2ETime, statsE2E);
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
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

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  cameras = await getVideoInputs();

  await setupDatGui(urlParams, cameras);

  statsModel = setupStats('stats1', '(Model FPS)      ');
  statsE2E = setupStats('stats2', '(End2End FPS)');

  camera = await Camera.setupCamera(STATE.camera, cameras);
  canvas.width = camera.canvas.width;
  canvas.height = camera.canvas.height;

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  segmenter = await createSegmenter();

  renderPrediction();
};

app();
