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

const videoWidth = isMobile() ? 480 : 640;
const videoHeight = isMobile() ? 480 : 480;

const STATE = {
  shaderType: 'Point lights',
  VisualizeDepth: 0,
  MinDepth: 0.3,
  MaxDepth: 0.9,
  LightDepth: 0.42,
  RayMarchingSteps: 128,
  GlobalBrightness: 0.6,
  LowerIntensity: 0.5,
  HigherIntensity: 1.8,
  MaxIntensity: 2.25,
  EnergyDecayFactor: 0.995,
  DepthWeight: 0.8,
  LightRadius: 0.07,
  LightSelfBrightness: 2.5,
  GlobalDarkness: 0.5,
  DepthCachedFrames: 4,
  LightsOffsetX: 0,
  LightsOffsetY: 0,
  LightsOffsetZ: 0,
  FirstLightColor: [204, 0.0, 0.0],
  SecondLightColor: [0.0, 204, 0.0],
  ThirdLightColor: [0.0, 0.0, 255],
};

const shaderUniforms = [
  ['VisualizeDepth', 0, 1, 'both'],
  ['MinDepth', 0, 1, 'both'],
  ['MaxDepth', 0, 1, 'both'],
  ['RayMarchingSteps', 1, 128, 'both'],
  ['DepthWeight', 0.1, 1, 'Sunbeam'],
  ['GlobalBrightness', 0.1, 3, 'Sunbeam'],
  ['MaxIntensity', 0.1, 5, 'Sunbeam'],
  ['EnergyDecayFactor', 0.98, 0.999, 'Sunbeam'],
  ['LowerIntensity', 0.1, 3, 'Sunbeam'],
  ['HigherIntensity', 1, 3, 'Sunbeam'],
  ['LightDepth', 0, 1, 'Sunbeam'],
  ['LightRadius', 0, 0.2, 'Sunbeam'],
  ['LightSelfBrightness', 0.1, 5, 'Sunbeam'],
  ['GlobalDarkness', 0, 1, 'Point lights'],
  ['LightsOffsetX', -1, 1, 'Point lights'],
  ['LightsOffsetY', -1, 1, 'Point lights'],
  ['LightsOffsetZ', -1, 1, 'Point lights'],
  ['FirstLightColor', 'Point lights'],
  ['SecondLightColor', 'Point lights'],
  ['ThirdLightColor', 'Point lights']
];