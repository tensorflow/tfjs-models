/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ConvolutionDefinition, mobileNetArchitectures} from './mobilenet';

const BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/';

export type Checkpoint = {
  url: string,
  architecture: ConvolutionDefinition[]
};

export const checkpoints: {[multiplier: number]: Checkpoint} = {
  1.0: {
    url: BASE_URL + 'posenet_mobilenet_100_partmap/',
    architecture: mobileNetArchitectures[100]
  },
  0.75: {
    url: BASE_URL + 'posenet_mobilenet_075_partmap/',
    architecture: mobileNetArchitectures[75]
  },
  0.5: {
    url: BASE_URL + 'posenet_mobilenet_050_partmap/',
    architecture: mobileNetArchitectures[50]
  },
  0.25: {
    url: BASE_URL + 'posenet_mobilenet_025_partmap/',
    architecture: mobileNetArchitectures[25]
  }
};
