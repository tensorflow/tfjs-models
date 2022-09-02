/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import * as params from './shared/params';
import {setupModelFolder} from './shared/option_panel';

export async function setupDatGui(urlParams) {
  const gui = new dat.GUI({width: 300});
  gui.domElement.id = 'gui';

  // The camera folder contains options for video settings.
  const cameraFolder = gui.addFolder('Camera');
  const fpsController = cameraFolder.add(params.STATE.camera, 'targetFPS');
  fpsController.onFinishChange((_) => {
    params.STATE.isTargetFPSChanged = true;
  });
  const sizeController = cameraFolder.add(
      params.STATE.camera, 'sizeOption', Object.keys(params.VIDEO_SIZE));
  sizeController.onChange(_ => {
    params.STATE.isSizeOptionChanged = true;
  });
  cameraFolder.open();

  return setupModelFolder(gui, urlParams);
}
