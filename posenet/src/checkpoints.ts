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

const BASE_URL = 'https://storage.googleapis.com/tfjs-models/weights/posenet/';
const RESNET50_BASE_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/posenet_resnet50/';

export type Checkpoint = {
  url: string,
  architecture: ConvolutionDefinition[]
};

export const checkpoints: {[multiplier: number]: Checkpoint} = {
  1.01: {
    url: BASE_URL + 'mobilenet_v1_101/',
    architecture: mobileNetArchitectures[100]
  },
  1.0: {
    url: BASE_URL + 'mobilenet_v1_100/',
    architecture: mobileNetArchitectures[100]
  },
  0.75: {
    url: BASE_URL + 'mobilenet_v1_075/',
    architecture: mobileNetArchitectures[75]
  },
  0.5: {
    url: BASE_URL + 'mobilenet_v1_050/',
    architecture: mobileNetArchitectures[50]
  }
};

// The ResNet50 models use the latest TensorFlow.js 1.0 model format. This is
// slightly different than the old format used by MobileNetV1 models. Thus we
// define a new data structure to contain them.
export function resNet50Checkpoint(
    resolution: number, stride: number, quantBytes: number):
    {[resolution: number]: {[multiplier: number]: string}} {
  const graphJson = `model-${resolution}x${resolution}-stride${stride}.json`;
  // quantBytes=4 corresponding to the non-quantized full-precision checkpoints.
  if (quantBytes == 4) {
    return RESNET50_BASE_URL + graphJson;
  } else {
    return RESNET50_BASE_URL + `quant${quantBytes}/` + graphJson;
  }
};
