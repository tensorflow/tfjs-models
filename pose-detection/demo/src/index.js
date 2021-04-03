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

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setupStats} from './stats_panel';

let detector, camera, stats;

async function createDetector(model) {
  switch (model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.MediapipeBlazepose:
      return posedetection.createDetector(
          STATE.model.model, {quantBytes: 4, upperBodyOnly: false});
    case posedetection.SupportedModels.MoveNet:
      const modelType =
          STATE.model[STATE.model.model].modelType == 'Lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(
          STATE.model.model, {modelType: modelType});
  }
}

async function checkGuiUpdate() {
  if (STATE.changeToTargetFPS || STATE.changeToSizeOption) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.changeToTargetFPS = null;
    STATE.changeToSizeOption = null;
  }

  if (STATE.changeToModel) {
    detector.dispose();
    detector = await createDetector(STATE.model.model);
    STATE.changeToModel = null;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  stats.begin();

  const poses = await detector.estimatePoses(
      camera.video, {maxPoses: 1, flipHorizontal: false});

  stats.end();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If changeToModel is non-null, the result is from an
  // old model, which shouldn't be rendered.
  if (poses.length > 0 && STATE.changeToModel == null) {
    const shouldScale =
        STATE.model.model === posedetection.SupportedModels.MediapipeBlazepose;

    camera.drawResult(poses[0], shouldScale);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  await renderResult();

  requestAnimationFrame(renderPrediction);
};

async function app() {
  await tf.setBackend('webgl');
  setupDatGui();
  stats = setupStats();
  camera = await Camera.setupCamera(STATE.camera);

  detector = await createDetector(STATE.model.model);

  renderPrediction();
};

app();
