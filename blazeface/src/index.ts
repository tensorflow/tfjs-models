/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tfconv from '@tensorflow/tfjs-converter';
import {BlazeFaceModel} from './face';

const BLAZEFACE_MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/facemesh_staging/facedetector_tfjs/model.json';

/**
 * Load blazeface.
 *
 * @param maxFaces The maximum number of faces returned by the model.
 * @param inputWidth The width of the input image.
 * @param inputHeight The height of the input image.
 * @param iouThreshold The threshold for deciding whether boxes overlap too
 * much.
 * @param scoreThreshold The threshold for deciding when to remove boxes based
 * on score.
 */
export async function load({
  maxFaces = 10,
  inputWidth = 128,
  inputHeight = 128,
  iouThreshold = 0.3,
  scoreThreshold = 0.75
} = {}) {
  const blazeface = await tfconv.loadGraphModel(BLAZEFACE_MODEL_URL);

  return new BlazeFaceModel(
      blazeface, inputWidth, inputHeight, maxFaces, iouThreshold,
      scoreThreshold);
}
