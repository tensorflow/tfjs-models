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
  const type = urlParams.get('type');
  switch (model) {
    case 'posenet':
      addPoseNetControllers(modelFolder);
      break;
    case 'movenet':
      addMoveNetControllers(modelFolder, type);
      break;
    case 'blazepose':
      addBlazePoseControllers(modelFolder, type);
      break;
    default:
      alert(`${urlParams.get('model')}`);
      break;
  }

  modelFolder.open();

  return gui;
}

// The MoveNet model config folder contains options for MoveNet config
// settings.
function addMoveNetControllers(modelFolder, type) {
  params.STATE.model = {
    model: posedetection.SupportedModels.MoveNet,
    ...params.MOVENET_CONFIG
  };

  let $type = type != null ? type : 'thunder';
  if ($type !== 'thunder' && $type !== 'lightning') {
    $type = 'thunder';
  }

  params.STATE.model.type = $type;

  const typeController =
      modelFolder.add(params.STATE.model, 'type', ['thunder', 'lightning']);
  typeController.onChange(type => {
    params.STATE.changeToModel = type;
  });

  modelFolder.add(params.STATE.model, 'scoreThreshold', 0, 1);
}

// The Blazepose model config folder contains options for Blazepose config
// settings.
function addBlazePoseControllers(modelFolder) {
  params.STATE.model = {
    model: posedetection.SupportedModels.MediapipeBlazepose,
    ...params.BLAZEPOSE_CONFIG
  };

  modelFolder.add(params.STATE.model, 'scoreThreshold', 0, 1);
}

// The PoseNet model config folder contains options for PoseNet config
// settings.
function addPoseNetControllers(modelFolder) {
  params.STATE.model = {
    model: posedetection.SupportedModels.PoseNet,
    ...params.POSENET_CONFIG
  };

  modelFolder.add(params.STATE.model, 'scoreThreshold', 0, 1);
}
