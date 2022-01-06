
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
import {BodyPixOutputStride} from './types';

/**
 * BodyPix supports using various convolution neural network models
 * (e.g. ResNet and MobileNetV1) as its underlying base model.
 * The following BaseModel interface defines a unified interface for
 * creating such BodyPix base models. Currently both MobileNet (in
 * ./mobilenet.ts) and ResNet (in ./resnet.ts) implements the BaseModel
 * interface. New base models that conform to the BaseModel interface can be
 * added to BodyPix.
 */
export abstract class BaseModel {
  constructor(
      protected readonly model: tfconv.GraphModel,
      public readonly outputStride: BodyPixOutputStride) {
    const inputShape =
        this.model.inputs[0].shape as [number, number, number, number];
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be equal to or -1`);
  }

  abstract preprocessInput(input: tf.Tensor3D): tf.Tensor3D;

  /**
   * Predicts intermediate Tensor representations.
   *
   * @param input The input RGB image of the base model.
   * A Tensor of shape: [`inputResolution`, `inputResolution`, 3].
   *
   * @return A dictionary of base model's intermediate predictions.
   * The returned dictionary should contains the following elements:
   * - heatmapScores: A Tensor3D that represents the keypoint heatmap scores.
   * - offsets: A Tensor3D that represents the offsets.
   * - displacementFwd: A Tensor3D that represents the forward displacement.
   * - displacementBwd: A Tensor3D that represents the backward displacement.
   * - segmentation: A Tensor3D that represents the segmentation of all
   * people.
   * - longOffsets: A Tensor3D that represents the long offsets used for
   * instance grouping.
   * - partHeatmaps: A Tensor3D that represents the body part segmentation.
   */
  predict(input: tf.Tensor3D): {
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
    segmentation: tf.Tensor3D,
    partHeatmaps: tf.Tensor3D,
    longOffsets: tf.Tensor3D,
    partOffsets: tf.Tensor3D
  } {
    return tf.tidy(() => {
      const asFloat = this.preprocessInput(tf.cast(input, 'float32'));
      const asBatch = tf.expandDims(asFloat, 0);
      const results = this.model.predict(asBatch) as tf.Tensor4D[];
      const results3d: tf.Tensor3D[] = results.map(y => tf.squeeze(y, [0]));
      const namedResults = this.nameOutputResults(results3d);

      return {
        heatmapScores: tf.sigmoid(namedResults.heatmap),
        offsets: namedResults.offsets,
        displacementFwd: namedResults.displacementFwd,
        displacementBwd: namedResults.displacementBwd,
        segmentation: namedResults.segmentation,
        partHeatmaps: namedResults.partHeatmaps,
        longOffsets: namedResults.longOffsets,
        partOffsets: namedResults.partOffsets
      };
    });
  }

  // Because MobileNet and ResNet predict() methods output a different order for
  // these values, we have a method that needs to be implemented to order them.
  abstract nameOutputResults(results: tf.Tensor3D[]): {
    heatmap: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
    segmentation: tf.Tensor3D,
    partHeatmaps: tf.Tensor3D,
    longOffsets: tf.Tensor3D,
    partOffsets: tf.Tensor3D
  };

  /**
   * Releases the CPU and GPU memory allocated by the model.
   */
  dispose() {
    this.model.dispose();
  }
}
