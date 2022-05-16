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
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {DepthEstimator} from '../depth_estimator';
import {toImageTensor, transformValueRange} from '../shared/calculators/image_utils';
import {toHTMLCanvasElementLossy} from '../shared/calculators/mask_util';
import {DepthEstimatorInput, DepthMap} from '../types';

import {segmentForeground} from './calculators/segment_foreground';
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

const TRANSFORM_INPUT_IMAGE = transformValueRange(0, 255, 0, 1);
const PORTRAIT_HEIGHT = 256;
const PORTRAIT_WIDTH = 192;
/**
 * ARPortraitDepth estimator class.
 */
class ARPortraitDepthEstimator implements DepthEstimator {
  constructor(
      private readonly segmenter: bodySegmentation.BodySegmenter,
      private readonly estimatorModel: tfconv.GraphModel) {}

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
   *       minDepth: The minimum depth value for the model to map to 0. Any
   * smaller depth values will also get mapped to 0.
   *
   *       maxDepth`: The maximum depth value for the model to map to 1. Any
   * larger depth values will also get mapped to 1.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   * from camera, the result has to flip horizontally.
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
    const [height, width] = image3d.shape;

    const segmentations = await this.segmenter.segmentPeople(image3d);

    // On a non-null input, SelfieSegmentation always returns one probability
    // mask.
    const segmentation = segmentations[0];
    const segmentationTensor = await segmentation.mask.toTensor();

    const depthTensor = tf.tidy(() => {
      const maskedImage = segmentForeground(image3d, segmentationTensor);
      segmentationTensor.dispose();

      // Normalizes the values from [0, 255] to [0, 1], same ranged used
      // during training.
      const imageNormalized =
          tf.add(
              tf.mul(maskedImage, TRANSFORM_INPUT_IMAGE.scale),
              // tslint:disable-next-line: no-unnecessary-type-assertion
              TRANSFORM_INPUT_IMAGE.offset) as tf.Tensor3D;

      const imageResized = tf.image.resizeBilinear(
          imageNormalized, [PORTRAIT_HEIGHT, PORTRAIT_WIDTH]);

      // Shape after expansion is [1, height, width, 3].
      const batchInput = tf.expandDims(imageResized);

      // Depth prediction (ouput shape is [1, height, width, 1]).
      const depth4D = this.estimatorModel.predict(batchInput) as tf.Tensor4D;

      // Normalize to user requirements.
      const depthTransform =
          transformValueRange(config.minDepth, config.maxDepth, 0, 1);

      // depth4D is roughly in [0,2] range, so half the scale factor to put it
      // in [0,1] range.
      const scale = depthTransform.scale / 2;
      const result =
          // tslint:disable-next-line: no-unnecessary-type-assertion
          tf.add(tf.mul(depth4D, scale), depthTransform.offset) as tf.Tensor3D;

      // Keep in [0,1] range.
      const resultClipped = tf.clipByValue(result, 0, 1);

      // Rescale to original input size.
      const resultResized =
          tf.image.resizeBilinear(resultClipped, [height, width]);

      // Remove channel dimension.
      // tslint:disable-next-line: no-unnecessary-type-assertion
      const resultSqueezed = tf.squeeze(resultResized, [0, 3]) as tf.Tensor2D;

      return resultSqueezed;
    });

    image3d.dispose();

    return new ARPortraitDepthMap(depthTensor);
  }

  dispose() {
    this.segmenter.dispose();
    this.estimatorModel.dispose();
  }

  reset() {
    this.segmenter.reset();
  }
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

  const depthModelFromTFHub = typeof config.depthModelUrl === 'string' &&
      (config.depthModelUrl.indexOf('https://tfhub.dev') > -1);

  const depthModel = await tfconv.loadGraphModel(
      config.depthModelUrl, {fromTFHub: depthModelFromTFHub});

  const segmenter = await bodySegmentation.createSegmenter(
      bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
      {runtime: 'tfjs', modelUrl: config.segmentationModelUrl});

  return new ARPortraitDepthEstimator(segmenter, depthModel);
}
