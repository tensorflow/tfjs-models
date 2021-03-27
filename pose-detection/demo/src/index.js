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

import * as posedetection from '@tensorflow-models/posedetection';
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
          STATE.model.model,
          {quantBytes: 4, upperBodyOnly: false, smoothConfig: STATE.blazePose});
  }
}

async function checkGuiUpdate() {
  let changes = false;

  if (STATE.changeToTargetFPS || STATE.changeToSizeOption) {
    changes = true;
    if (STATE.changeToTargetFPS) {
      STATE.camera.targetFPS = STATE.changeToTargetFPS;
      STATE.changeToTargetFPS = null;
    }

    if (STATE.changeToSizeOption) {
      STATE.camera.sizeOption = STATE.changeToSizeOption;
      STATE.changeToSizeOption = null;
    }

    camera = await Camera.setupCamera(STATE.camera);
  }

  if (STATE.changeToModel) {
    changes = true;
    STATE.model.model = STATE.changeToModel;
    STATE.changeToModel = null;

    detector.dispose();
    detector = await createDetector(STATE.model.model);
  }

  if (STATE.changeToWindowSize || STATE.changeToVelocityScale) {
    changes = true;
    console.log('there change');
    if (STATE.changeToWindowSize) {
      console.log('change to window size');
      STATE.blazePose.windowSize = STATE.changeToWindowSize;
      STATE.changeToWindowSize = null;
    }

    if (STATE.changeToVelocityScale) {
      STATE.blazePose.velocityScale = STATE.changeToVelocityScale;
      STATE.changeToVelocityScale = null;
    }

    detector.dispose();
    detector = await createDetector(STATE.model.model);
  }

  if (changes) {
    await tf.nextFrame();
  }
}

async function renderResult() {
  if (camera.video.currentTime !== camera.lastVideoTime) {
    camera.lastVideoTime = camera.video.currentTime;

    stats.begin();

    const poses = await detector.estimatePoses(
        camera.video, {maxPoses: 1, flipHorizontal: false, enableSmooth: true});

    stats.end();

    camera.drawCtx();

    if (poses.length > 0) {
      const shouldScale = STATE.model.model ===
          posedetection.SupportedModels.MediapipeBlazepose;

      camera.drawResult(poses[0], shouldScale);
    }
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
