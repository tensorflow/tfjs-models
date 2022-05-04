/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import * as tf from '@tensorflow/tfjs-core';

import {DepthEstimator} from '../depth_estimator';
import {getImageSize, toImageTensor, transformValueRange} from '../shared/calculators/image_utils';
import {toHTMLCanvasElementLossy} from '../shared/calculators/mask_util';
import {DepthEstimatorInput, DepthMap} from '../types';

import {validateEstimationConfig, validateModelConfig} from './estimator_utils';
import {ARPortraitDepthEstimationConfig, ARPortraitDepthModelConfig} from './types';

class ARPortraitDepthMap implements DepthMap {
  constructor(private depthTensor: tf.Tensor2D) {}

  async toCanvasImageSource() {
    return toHTMLCanvasElementLossy(this.depthTensor);
  }

  async toArray() {
    return this.depthTensor.arraySync();
  }

  async toTensor() {
    return this.depthTensor;
  }

  getUnderlyingType() {
    return 'tensor' as const ;
  }
}

/**
 * ARPortraitDepth estimator class.
 */
class ARPortraitDepthEstimator implements DepthEstimator {
  constructor(
      private readonly estimatorModel: tfconv.GraphModel,
      private readonly minDepth: number, private readonly maxDepth: number) {}

  /**
   * Estimates depth for an image or video frame.
   *
   * It returns a depth map of the same number of values as input pixels.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return `DepthMap`.
   */
  async estimateDepth(
      image: DepthEstimatorInput,
      estimationConfig?: ARPortraitDepthEstimationConfig): Promise<DepthMap> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return null;
    }

    const image3d = tf.tidy(() => {
      let imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (config.flipHorizontal) {
        const batchAxis = 0;
        imageTensor = tf.squeeze(
            tf.image.flipLeftRight(
                // tslint:disable-next-line: no-unnecessary-type-assertion
                tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D),
            [batchAxis]);
      }
      return imageTensor;
    });

    const {height, width} = getImageSize(image3d);

    // Normalizes the values from [0, 255] to [0, 1], same ranged used
    // during training.
    const transformInput = transformValueRange(0, 255, 0, 1);
    this.minDepth;
    this.maxDepth;

    const depthTensor = tf.tidy(() => {
      const imageResized = tf.image.resizeBilinear(image3d, [256, 192]);
      // Masks the image
      const imageNormalized = tf.add(
          tf.mul(imageResized, transformInput.scale), transformInput.offset);

      // Runs the model.
      const batchInput = tf.expandDims(imageNormalized);

      // Depth prediction.
      const depth4D = this.estimatorModel.predict(batchInput) as tf.Tensor4D;

      // Remove batch dimension.
      const depth3D = tf.squeeze(depth4D, [0]);

      // Output is roughly in [0,2] range, normalize to [0,1]
      const depthNormalized = tf.div(depth3D, 2);

      // Normalize to user requirements.
      const result = tf.div(
                         tf.sub(depthNormalized, this.minDepth),
                         this.maxDepth - this.minDepth) as tf.Tensor3D;

      // Keep in [0,1] range.
      const resultClipped = tf.clipByValue(result, 0, 1);

      // Rescale to original input size.
      const resultResized =
          tf.image.resizeBilinear(resultClipped, [height, width]);

      // Remove channel dimension.
      const resultSqueezed = tf.squeeze(resultResized, [2]);

      return resultSqueezed;
    });

    tf.dispose(image3d);

    return new ARPortraitDepthMap(depthTensor as tf.Tensor2D);
  }

  dispose() {
    this.estimatorModel.dispose();
  }

  reset() {}
}

/**
 * Loads the ARPortraitDepth model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the ARPortraitDepth loading process. Please find more details of each
 * parameters in the documentation of the `ARPortraitDepthModelConfig`
 * interface.
 */
export async function load(modelConfig: ARPortraitDepthModelConfig):
    Promise<DepthEstimator> {
  const config = validateModelConfig(modelConfig);

  const modelFromTFHub = typeof config.modelUrl === 'string' &&
      (config.modelUrl.indexOf('https://tfhub.dev') > -1);

  const model =
      await tfconv.loadGraphModel(config.modelUrl, {fromTFHub: modelFromTFHub});

  return new ARPortraitDepthEstimator(model, config.minDepth, config.maxDepth);
}
