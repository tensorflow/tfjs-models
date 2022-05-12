/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var config = {
  cameraRadius: 0.4,
  cameraSpeed: 0.005,
  cameraX: 0.0,
  cameraY: 0.0,
  cameraZ: 9.5,
  cameraLookAtX: 0,
  cameraLookAtY: 0,
  cameraLookAtZ: -1,
  minDepth: 0.2,
  maxDepth: 0.9,
  imageScale: 0.025,
  depthScale: -5.0,
  autoAnimation: true,
  showBackgroundPic: true,
  backgroundScale: 1.0,
  backgroundDepth: -2.7,
  backgroundColor: '#050505',
};
var capturer;
var capturerInitialTheta;
var updateDepthCallback;
