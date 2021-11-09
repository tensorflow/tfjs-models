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

import * as tf from '@tensorflow/tfjs-core';

import {BaseModel} from './base_model';

const imageNetMean = [-123.15, -115.90, -103.06];

export class ResNet extends BaseModel {
  preprocessInput(input: tf.Tensor3D): tf.Tensor3D {
    return tf.add(input, imageNetMean);
  }

  nameOutputResults(results: tf.Tensor3D[]) {
    const [displacementFwd, displacementBwd, offsets, heatmap] = results;
    return {offsets, heatmap, displacementFwd, displacementBwd};
  }
}
