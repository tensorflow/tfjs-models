/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

const RESNET50_BASE_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/';
const MOBILENET_BASE_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/';

// The BodyPix 2.0 ResNet50 models use the latest TensorFlow.js 1.0 model
// format.
export function resNet50SavedModel(stride: number, quantBytes: number): string {
  const graphJson = `model-stride${stride}.json`;
  // quantBytes=4 corresponding to the non-quantized full-precision SavedModel.
  if (quantBytes === 4) {
    return RESNET50_BASE_URL + `float/` + graphJson;
  } else {
    return RESNET50_BASE_URL + `quant${quantBytes}/` + graphJson;
  }
}

// The BodyPix 2.0 MobileNetV1 models use the latest TensorFlow.js 1.0 model
// format.
export function mobileNetSavedModel(
    stride: number, multiplier: number, quantBytes: number): string {
  const toStr: {[key: number]: string} = {1.0: '100', 0.75: '075', 0.50: '050'};
  const graphJson = `model-stride${stride}.json`;
  // quantBytes=4 corresponding to the non-quantized full-precision SavedModel.
  if (quantBytes === 4) {
    return MOBILENET_BASE_URL + `float/${toStr[multiplier]}/` + graphJson;
  } else {
    return MOBILENET_BASE_URL + `quant${quantBytes}/${toStr[multiplier]}/` +
        graphJson;
  }
}
