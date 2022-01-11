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
import * as mpSelfieSegmentation from '@mediapipe/selfie_segmentation';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {setupStats} from './shared/stats_panel';
import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './shared/params';
import {setBackendAndEnvFlags} from './shared/util';

let segmenter, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const statusElement = document.getElementById('status');
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
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`,
          enableSegmentation: true,
          smoothSegmentation: true
        });
      } else if (runtime === 'tfjs') {
        return poseDetection.createDetector(
          STATE.model, {runtime, modelType: STATE.modelConfig.type,
                        enableSegmentation: true, smoothSegmentation: true});
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
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@${mpSelfieSegmentation.VERSION}`
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
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged || STATE.isVisChanged) {
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

function beginEstimateSegmentationStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateSegmentationStats() {
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
  let segmentation = null;

  // Segmenter can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (segmenter != null) {
    // FPS only counts the time it takes to finish segmentPeople.
    beginEstimateSegmentationStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      if (segmenter.segmentPeople != null) {
        segmentation = await segmenter.segmentPeople(
          camera.video,
          {flipHorizontal: false, multiSegmentation: false, segmentBodyParts: true,
            segmentationThreshold: STATE.visualization.foregroundThreshold});
      } else {
        segmentation = await segmenter.estimatePoses(camera.video, {flipHorizontal: false});
        segmentation = segmentation.map(singleSegmentation => singleSegmentation.segmentation);
      }
    } catch (error) {
      segmenter.dispose();
      segmenter = null;
      alert(error);
    }

    endEstimateSegmentationStats();
  }

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (segmentation && segmentation.length > 0 && !STATE.isModelChanged) {
    const vis = STATE.modelConfig.visualization;
    const options = STATE.visualization;
    if (vis === 'binaryMask') {
      const data = await bodySegmentation.toBinaryMask(segmentation, {r: 0, g: 0, b: 0, a: 0}, {r: 0, g: 0, b: 0, a:255}, false, options.foregroundThreshold);
      await bodySegmentation.drawMask(canvas, camera.video, data, options.maskOpacity, options.maskBlur);
    } else if (vis === 'coloredMask') {
      const data = await bodySegmentation.toColoredMask(segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor, {r: 0, g: 0, b: 0, a:255}, options.foregroundThreshold);
      await bodySegmentation.drawMask(canvas, camera.video, data, options.maskOpacity, options.maskBlur);
    } else if (vis === 'pixelatedMask') {
      const data = await bodySegmentation.toColoredMask(segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor, {r: 0, g: 0, b: 0, a:255}, options.foregroundThreshold);
      await bodySegmentation.drawPixelatedMask(canvas, camera.video, data, options.maskOpacity, options.maskBlur, false, options.pixelCellWidth);
    } else if (vis === 'bokehEffect') {
      await bodySegmentation.drawBokehEffect(canvas, camera.video, segmentation, options.foregroundThreshold, options.backgroundBlur, options.edgeBlur);
    } else if (vis === 'blurFace') {
      await bodySegmentation.blurBodyPart(canvas, camera.video, segmentation, [0,1], options.foregroundThreshold, options.backgroundBlur, options.edgeBlur);
    } else {
      camera.drawFromVideo(ctx);
    }
  }
  camera.drawToCanvas(canvas);

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
  let segmentation = null;

  if (runtime === 'tfjs') {
    const warmUpTensor =
        tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    if (segmenter.segmentPeople != null) {
      segmentation = await segmenter.segmentPeople(
        warmUpTensor,
        {flipHorizontal: false, multiSegmentation: false, segmentBodyParts: true,
          segmentationThreshold: STATE.visualization.foregroundThreshold});
    } else {
      segmentation = await segmenter.estimatePoses(warmUpTensor, {flipHorizontal: false});
      segmentation = segmentation.map(singleSegmentation => singleSegmentation.segmentation);
    }
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
  segmenter = await createSegmenter();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  checkUpdate();
};

app();
