/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

import {BaseModel} from './base_model';

export class MobileNet extends BaseModel {
  preprocessInput(input: tf.Tensor3D): tf.Tensor3D {
    // Normalize the pixels [0, 255] to be between [-1, 1].
    return tf.tidy(() => tf.div(input, 127.5).sub(1.0));
  }

  nameOutputResults(results: tf.Tensor3D[]) {
    const [
      offsets,
      segmentation,
      partHeatmaps,
      longOffsets,
      heatmap,
      displacementFwd,
      displacementBwd,
      partOffsets,
  ] = results;
    return {
      offsets,
      segmentation,
      partHeatmaps,
      longOffsets,
      heatmap,
      displacementFwd,
      displacementBwd,
      partOffsets
    };
  }
}
