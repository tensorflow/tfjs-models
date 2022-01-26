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
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import * as params from './shared/params';

/**
 * Records each flag's default value under the runtime environment and is a
 * constant in runtime.
 */
let TUNABLE_FLAG_DEFAULT_VALUE_MAP;

const stringValueMap = {};

function toCameraOptions(cameras) {
  const result = {default: null};

  cameras.forEach(camera => {
    result[camera.label] = camera.label;
  })

  return result;
}

export async function setupDatGui(urlParams, cameras) {
  const gui = new dat.GUI({width: 300});
  gui.domElement.id = 'gui';

  // The camera folder contains options for video settings.
  const cameraFolder = gui.addFolder('Camera');
  const fpsController = cameraFolder.add(params.STATE.camera, 'targetFPS');
  fpsController.onFinishChange((_) => {
    params.STATE.isCameraChanged = true;
  });
  const sizeController = cameraFolder.add(
      params.STATE.camera, 'sizeOption', Object.keys(params.VIDEO_SIZE));
  sizeController.onChange(_ => {
    params.STATE.isCameraChanged = true;
  });
  const cameraOptions = toCameraOptions(cameras);
  params.STATE.camera.cameraSelector = cameraOptions['default'];
  const cameraSelectorController =
      cameraFolder.add(params.STATE.camera, 'cameraSelector', cameraOptions);
  cameraSelectorController.onChange(_ => {
    params.STATE.isCameraChanged = true;
  });
  cameraFolder.open();

  // The fps display folder contains options for video settings.
  const fpsDisplayFolder = gui.addFolder('FPS Display');
  fpsDisplayFolder.add(params.STATE.fpsDisplay, 'mode', ['model', 'e2e']);
  fpsDisplayFolder.open();

  // The model folder contains options for model selection.
  const modelFolder = gui.addFolder('Model');

  const model = urlParams.get('model');
  let type = urlParams.get('type');

  switch (model) {
    case 'blazepose':
      params.STATE.model = poseDetection.SupportedModels.BlazePose;
      break;
    case 'bodypix':
      params.STATE.model = bodySegmentation.SupportedModels.BodyPix;
      break;
    case 'selfie_segmentation':
      params.STATE.model =
          bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
      if (type !== 'general' && type !== 'landscape') {
        // Nulify invalid value.
        type = null;
      }
      break;
    default:
      alert(`${urlParams.get('model')}`);
      break;
  }

  const modelNames = Object.values(bodySegmentation.SupportedModels);
  modelNames.push(poseDetection.SupportedModels.BlazePose);
  const modelController = modelFolder.add(params.STATE, 'model', modelNames);

  modelController.onChange(_ => {
    params.STATE.isModelChanged = true;
    const visSelector = showModelConfigs(modelFolder);
    showBackendConfigs(backendFolder);
    visSelector.onChange(async mode => {
      params.STATE.isVisChanged = true;
      showVisualizationSettings(visFolder, mode);
    });
    showVisualizationSettings(
        visFolder, params.STATE.modelConfig.visualization);
  });

  const visSelector = showModelConfigs(modelFolder, type);

  modelFolder.open();

  const backendFolder = gui.addFolder('Backend');

  showBackendConfigs(backendFolder);

  backendFolder.open();

  const visFolder = gui.addFolder('Visualization');

  showVisualizationSettings(visFolder, 'binaryMask');

  visFolder.open();

  visSelector.onChange(async mode => {
    params.STATE.isVisChanged = true;
    showVisualizationSettings(visFolder, mode);
  });

  return gui;
}

async function showBackendConfigs(folderController) {
  // Clean up backend configs for the previous model.
  const fixedSelectionCount = 0;
  while (folderController.__controllers.length > fixedSelectionCount) {
    folderController.remove(
        folderController
            .__controllers[folderController.__controllers.length - 1]);
  }
  const backends = params.MODEL_BACKEND_MAP[params.STATE.model];
  // The first element of the array is the default backend for the model.
  params.STATE.backend = backends[0];
  const backendController =
      folderController.add(params.STATE, 'backend', backends);
  backendController.name('runtime-backend');
  backendController.onChange(async backend => {
    params.STATE.isBackendChanged = true;
    await showFlagSettings(folderController, backend);
  });
  await showFlagSettings(folderController, params.STATE.backend);
}

function showModelConfigs(folderController, type) {
  // Clean up model configs for the previous model.
  // The first constroller under the `folderController` is the model
  // selection.
  const fixedSelectionCount = 1;
  while (folderController.__controllers.length > fixedSelectionCount) {
    folderController.remove(
        folderController
            .__controllers[folderController.__controllers.length - 1]);
  }

  switch (params.STATE.model) {
    case poseDetection.SupportedModels.BlazePose:
      return addBlazePoseControllers(folderController, type);
    case bodySegmentation.SupportedModels.BodyPix:
      return addBodyPixControllers(folderController);
    case bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation:
      return addSelfieSegmentationControllers(folderController, type);
    default:
      alert(`Model ${params.STATE.model} is not supported.`);
  }
}

function showVisualizationSettings(folderController, vis) {
  // Clean up visualization configs for the previous mode.
  while (folderController.__controllers.length > 0) {
    folderController.remove(
        folderController
            .__controllers[folderController.__controllers.length - 1]);
  }

  folderController.add(
      params.STATE.visualization, 'foregroundThreshold', 0.0, 1.0);

  if (vis === 'binaryMask') {
    folderController.add(params.STATE.visualization, 'maskOpacity', 0.0, 1.0);
    folderController.add(params.STATE.visualization, 'maskBlur')
        .min(1)
        .max(20)
        .step(1);
  } else if (vis === 'coloredMask') {
    folderController.add(params.STATE.visualization, 'maskOpacity', 0.0, 1.0);
    folderController.add(params.STATE.visualization, 'maskBlur')
        .min(1)
        .max(20)
        .step(1);
  } else if (vis === 'pixelatedMask') {
    folderController.add(params.STATE.visualization, 'maskOpacity', 0.0, 1.0);
    folderController.add(params.STATE.visualization, 'maskBlur')
        .min(0)
        .max(20)
        .step(1);
    folderController.add(params.STATE.visualization, 'pixelCellWidth')
        .min(1)
        .max(50)
        .step(1);
  } else if (vis === 'bokehEffect') {
    folderController.add(params.STATE.visualization, 'backgroundBlur')
        .min(1)
        .max(20)
        .step(1);
    folderController.add(params.STATE.visualization, 'edgeBlur')
        .min(0)
        .max(20)
        .step(1);
  } else if (vis === 'blurFace') {
    folderController.add(params.STATE.visualization, 'backgroundBlur')
        .min(1)
        .max(20)
        .step(1);
    folderController.add(params.STATE.visualization, 'edgeBlur')
        .min(0)
        .max(20)
        .step(1);
  }
}

// The MediaPipeHands model config folder contains options for MediaPipeHands
// config settings.
function addSelfieSegmentationControllers(modelConfigFolder, type) {
  params.STATE.modelConfig = {...params.SELFIE_SEGMENTATION_CONFIG};
  params.STATE.modelConfig.type = type != null ? type : 'general';

  const typeController = modelConfigFolder.add(
      params.STATE.modelConfig, 'type', ['general', 'landscape']);
  typeController.onChange(_ => {
    // Set isModelChanged to true, so that we don't render any result during
    // changing models.
    params.STATE.isModelChanged = true;
  });

  const visSelector = modelConfigFolder.add(
      params.STATE.modelConfig, 'visualization', ['binaryMask', 'bokehEffect']);
  return visSelector;
}

// The BodyPix model config folder contains options for BodyPix config
// settings.
function addBodyPixControllers(modelConfigFolder) {
  params.STATE.modelConfig = {...params.BODY_PIX_CONFIG};

  const controllers = [];
  controllers.push(modelConfigFolder.add(
      params.STATE.modelConfig, 'architecture', ['ResNet50', 'MobileNetV1']));
  controllers.push(
      modelConfigFolder.add(params.STATE.modelConfig, 'outputStride', [8, 16]));
  controllers.push(modelConfigFolder.add(
      params.STATE.modelConfig, 'multiplier', [0.50, 0.75, 1.0]));
  controllers.push(
      modelConfigFolder.add(params.STATE.modelConfig, 'quantBytes', [1, 2, 4]));

  for (const controller of controllers) {
    controller.onChange(_ => {
      // Set isModelChanged to true, so that we don't render any result during
      // changing models.
      params.STATE.isModelChanged = true;
    });
  }

  const visSelector =
      modelConfigFolder.add(params.STATE.modelConfig, 'visualization', [
        'binaryMask', 'coloredMask', 'pixelatedMask', 'bokehEffect', 'blurFace'
      ]);
  return visSelector;
}

// The BlazePose model config folder contains options for BlazePose config
// settings.
function addBlazePoseControllers(modelConfigFolder, type) {
  params.STATE.modelConfig = {...params.BLAZE_POSE_CONFIG};
  params.STATE.modelConfig.type = type != null ? type : 'full';

  const typeController = modelConfigFolder.add(
      params.STATE.modelConfig, 'type', ['lite', 'full', 'heavy']);
  typeController.onChange(_ => {
    // Set isModelChanged to true, so that we don't render any result during
    // changing models.
    params.STATE.isModelChanged = true;
  });

  const visSelector = modelConfigFolder.add(
      params.STATE.modelConfig, 'visualization', ['binaryMask', 'bokehEffect']);
  return visSelector;
}

/**
 * Query all tunable flags' default value and populate `STATE.flags` with them.
 */
async function initDefaultValueMap() {
  // Clean up the cache to query tunable flags' default values.
  TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
  params.STATE.flags = {};
  for (const backend in params.BACKEND_FLAGS_MAP) {
    for (let index = 0; index < params.BACKEND_FLAGS_MAP[backend].length;
         index++) {
      const flag = params.BACKEND_FLAGS_MAP[backend][index];
      TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag] = await tf.env().getAsync(flag);
    }
  }

  // Initialize STATE.flags with tunable flags' default values.
  for (const flag in TUNABLE_FLAG_DEFAULT_VALUE_MAP) {
    if (params.BACKEND_FLAGS_MAP[params.STATE.backend].indexOf(flag) > -1) {
      params.STATE.flags[flag] = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
    }
  }
}

/**
 * Heuristically determine flag's value range based on flag's default value.
 *
 * Assume that the flag's default value has already chosen the best option for
 * the runtime environment, so users can only tune the flag value downwards.
 *
 * For example, if the default value of `WEBGL_RENDER_FLOAT32_CAPABLE` is false,
 * the tunable range is [false]; otherwise, the tunable range is [true. false].
 *
 * @param {string} flag
 */
function getTunableRange(flag) {
  const defaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
    return [false, true];
  } else if (flag === 'WEBGL_VERSION') {
    const tunableRange = [];
    for (let value = 1; value <= defaultValue; value++) {
      tunableRange.push(value);
    }
    return tunableRange;
  } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
    const tunableRange = [-1];
    for (let value = 0; value <= 2; value += 0.25) {
      tunableRange.push(value);
    }
    return tunableRange;
  } else if (typeof defaultValue === 'boolean') {
    return defaultValue ? [false, true] : [false];
  } else if (params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag] != null) {
    return params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag];
  } else {
    return [defaultValue];
  }
}

/**
 * Show flag settings for the given backend under the UI element of
 * `folderController`.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
function showBackendFlagSettings(folderController, backendName) {
  const tunableFlags = params.BACKEND_FLAGS_MAP[backendName];
  for (let index = 0; index < tunableFlags.length; index++) {
    const flag = tunableFlags[index];
    const flagName = params.TUNABLE_FLAG_NAME_MAP[flag] || flag;

    // When tunable (bool) and range (array) attributes of `flagRegistry` is
    // implemented, we can apply them to here.
    const flagValueRange = getTunableRange(flag);
    // Heuristically consider a flag with at least two options as tunable.
    if (flagValueRange.length < 2) {
      console.warn(
          `The ${flag} is considered as untunable, ` +
          `because its value range is [${flagValueRange}].`);
      continue;
    }

    let flagController;
    if (typeof flagValueRange[0] === 'boolean') {
      // Show checkbox for boolean flags.
      flagController = folderController.add(params.STATE.flags, flag);
    } else {
      // Show dropdown for other types of flags.
      flagController =
          folderController.add(params.STATE.flags, flag, flagValueRange);

      // Because dat.gui always casts dropdown option values to string, we need
      // `stringValueMap` and `onFinishChange()` to recover the value type.
      if (stringValueMap[flag] == null) {
        stringValueMap[flag] = {};
        for (let index = 0; index < flagValueRange.length; index++) {
          const realValue = flagValueRange[index];
          const stringValue = String(flagValueRange[index]);
          stringValueMap[flag][stringValue] = realValue;
        }
      }
      flagController.onFinishChange(stringValue => {
        params.STATE.flags[flag] = stringValueMap[flag][stringValue];
      });
    }
    flagController.name(flagName).onChange(() => {
      params.STATE.isFlagChanged = true;
    });
  }
}

/**
 * Set up flag settings under the UI element of `folderController`:
 * - If it is the first call, initialize the flags' default value and show flag
 * settings for both the general and the given backend.
 * - Else, clean up flag settings for the previous backend and show flag
 * settings for the new backend.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
async function showFlagSettings(folderController, backendName) {
  await initDefaultValueMap();

  // Clean up flag settings for the previous backend.
  // The first constroller under the `folderController` is the backend
  // setting.
  const fixedSelectionCount = 1;
  while (folderController.__controllers.length > fixedSelectionCount) {
    folderController.remove(
        folderController
            .__controllers[folderController.__controllers.length - 1]);
  }

  // Show flag settings for the new backend.
  showBackendFlagSettings(folderController, backendName);
}
