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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {BaseModel, BodyPixOutputStride} from './body_pix_model';

export type MobileNetMultiplier = 0.50|0.75|1.0;

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32') {
      input = input.toFloat();
    }
    return input;
  });
}

function processInput(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    // Normalize the pixels [0, 255] to be between [-1, 1].
    return tf.div(input, 127.5).sub(1.0);
  });
}

export class MobileNet implements BaseModel {
  readonly model: tfconv.GraphModel;
  readonly outputStride: BodyPixOutputStride;

  constructor(model: tfconv.GraphModel, outputStride: BodyPixOutputStride) {
    this.model = model;
    const inputShape =
        this.model.inputs[0].shape as [number, number, number, number];
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be -1`);
    this.outputStride = outputStride;
  }

  predict(input: tf.Tensor3D): {[key: string]: tf.Tensor3D} {
    return tf.tidy(() => {
      const asFloat = processInput(toFloatIfInt(input));
      const asBatch = asFloat.expandDims(0);
      const [
          offsets4d,
          segmentation4d,
          partHeatmaps4d,
          longOffsets4d,
          heatmaps4d,
          displacementFwd4d,
          displacementBwd4d,
          partOffsets4d,
      ] = this.model.predict(asBatch) as tf.Tensor[];

      const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze() as tf.Tensor3D;
      const displacementFwd = displacementFwd4d.squeeze([0]) as tf.Tensor3D;
      const displacementBwd = displacementBwd4d.squeeze([0]) as tf.Tensor3D;
      const segmentation = segmentation4d.squeeze([0]) as tf.Tensor3D;
      const partHeatmaps = partHeatmaps4d.squeeze([0]) as tf.Tensor3D;
      const longOffsets = longOffsets4d.squeeze([0]) as tf.Tensor3D;
      const partOffsets = partOffsets4d.squeeze([0]) as tf.Tensor3D;

      return {
        heatmapScores,
        offsets: offsets as tf.Tensor3D,
        displacementFwd: displacementFwd as tf.Tensor3D,
        displacementBwd: displacementBwd as tf.Tensor3D,
        segmentation: segmentation as tf.Tensor3D,
        partHeatmaps: partHeatmaps as tf.Tensor3D,
        longOffsets: longOffsets as tf.Tensor3D,
        partOffsets: partOffsets as tf.Tensor3D
      };
    });
  }

  dispose() {
    this.model.dispose();
  }
}
