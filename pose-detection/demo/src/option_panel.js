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
import * as posedetection from '@tensorflow-models/pose-detection';

import * as params from './params';

export function setupDatGui(urlParams) {
  const gui = new dat.GUI({width: 300});

  // The camera folder contains options for video settings.
  const cameraFolder = gui.addFolder('Camera');
  const fpsController = cameraFolder.add(params.STATE.camera, 'targetFPS');
  fpsController.onFinishChange((targetFPS) => {
    params.STATE.changeToTargetFPS = +targetFPS;
  });
  const sizeController = cameraFolder.add(
      params.STATE.camera, 'sizeOption', Object.keys(params.VIDEO_SIZE));
  sizeController.onChange(option => {
    params.STATE.changeToSizeOption = option;
  });
  cameraFolder.open();

  // The model folder contains options for model selection.
  const modelFolder = gui.addFolder('Model');

  const model = urlParams.get('model');
  let type = urlParams.get('type');

  let modelConfigFolder;

  switch (model) {
    case 'posenet':
      params.STATE.model = posedetection.SupportedModels.PoseNet;
      break;
    case 'movenet':
      params.STATE.model = posedetection.SupportedModels.MoveNet;
      if (type !== 'lightning' && type !== 'thunder') {
        // Nulify invalid value.
        type = null;
      }
      break;
    case 'blazepose':
      params.STATE.model = type === 'upperbody' ?
          posedetection.SupportedModels.MediapipeBlazeposeUpperBody :
          posedetection.SupportedModels.MediapipeBlazeposeFullBody;
      if (type !== 'fullbody' && type !== 'upperbody') {
        // Nulify invalid value.
        type = null;
      }
      break;
    default:
      alert(`${urlParams.get('model')}`);
      break;
  }

  const modelController = modelFolder.add(
      params.STATE, 'model', Object.values(posedetection.SupportedModels));

  modelController.onChange(model => {
    params.STATE.changeToModel = model;

    // We don't pass in type, so that it will use default type when switching
    // models.
    modelConfigFolder = updateModelConfigFolder(gui, model, modelConfigFolder);

    modelConfigFolder.open();
  });

  modelFolder.open();

  // For initialization, pass in type from url.
  modelConfigFolder =
      updateModelConfigFolder(gui, params.STATE.model, modelConfigFolder, type);

  modelConfigFolder.open();

  return gui;
}

function updateModelConfigFolder(gui, model, modelConfigFolder, type) {
  if (modelConfigFolder != null) {
    gui.removeFolder(modelConfigFolder);
  }

  const newModelConfigFolder = gui.addFolder('Model Config');

  switch (model) {
    case posedetection.SupportedModels.PoseNet:
      addPoseNetControllers(newModelConfigFolder);
      break;
    case posedetection.SupportedModels.MoveNet:
      addMoveNetControllers(newModelConfigFolder, type);
      break;
    case posedetection.SupportedModels.MediapipeBlazeposeUpperBody:
    case posedetection.SupportedModels.MediapipeBlazeposeFullBody:
      addBlazePoseControllers(newModelConfigFolder);
      break;
    default:
      alert(`Model ${model} is not supported.`);
  }

  return newModelConfigFolder;
}

// The PoseNet model config folder contains options for PoseNet config
// settings.
function addPoseNetControllers(modelConfigFolder) {
  params.STATE.modelConfig = {...params.POSENET_CONFIG};
  modelConfigFolder.add(params.STATE.modelConfig, 'scoreThreshold', 0, 1);
}

// The MoveNet model config folder contains options for MoveNet config
// settings.
function addMoveNetControllers(modelConfigFolder, type) {
  params.STATE.modelConfig = {...params.MOVENET_CONFIG};
  params.STATE.modelConfig.type = type != null ? type : 'thunder';

  const typeController = modelConfigFolder.add(
      params.STATE.modelConfig, 'type', ['thunder', 'lightning']);
  typeController.onChange(_ => {
    // Set changeToModel to non-null, so that we don't render any result when
    // changeToModel is non-null.
    params.STATE.changeToModel = params.STATE.model;
  });

  modelConfigFolder.add(params.STATE.modelConfig, 'scoreThreshold', 0, 1);
}

// The Blazepose model config folder contains options for Blazepose config
// settings.
function addBlazePoseControllers(modelConfigFolder) {
  params.STATE.modelConfig = {...params.BLAZEPOSE_CONFIG};
  modelConfigFolder.add(params.STATE.modelConfig, 'scoreThreshold', 0, 1);
}
