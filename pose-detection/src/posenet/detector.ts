/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {convertImageToTensor} from '../calculators/convert_image_to_tensor';
import {getImageSize} from '../calculators/image_utils';
import {shiftImageValue} from '../calculators/shift_image_value';

import {BasePoseDetector, PoseDetector} from '../pose_detector';
import {InputResolution, Pose, PoseDetectorInput} from '../types';
import {decodeMultiplePoses} from './calculators/decode_multiple_poses';
import {decodeSinglePose} from './calculators/decode_single_pose';
import {flipPosesHorizontal} from './calculators/flip_poses';
import {scalePoses} from './calculators/scale_poses';

import {MOBILENET_V1_CONFIG, RESNET_MEAN, SINGLE_PERSON_ESTIMATION_CONFIG} from './constants';
import {assertValidOutputStride, assertValidResolution, validateEstimationConfig, validateModelConfig} from './detector_utils';
import {getValidInputResolutionDimensions, mobileNetCheckpoint, resNet50Checkpoint} from './load_utils';
import {PoseNetArchitecture, PoseNetEstimationConfig, PosenetModelConfig, PoseNetOutputStride} from './types';

/**
 * PoseNet detector class.
 */
export class PosenetDetector extends BasePoseDetector {
  private readonly inputResolution: InputResolution;
  private readonly architecture: PoseNetArchitecture;
  private readonly outputStride: PoseNetOutputStride;

  private maxPoses: number;

  // Should not be called outside.
  private constructor(
      private readonly posenetModel: tfconv.GraphModel,
      config: PosenetModelConfig) {
    super();

    // validate params.
    const inputShape =
        this.posenetModel.inputs[0].shape as [number, number, number, number];
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be equal to or -1`);

    const validInputResolution = getValidInputResolutionDimensions(
        config.inputResolution, config.outputStride);

    assertValidOutputStride(config.outputStride);
    assertValidResolution(validInputResolution, config.outputStride);

    this.inputResolution = validInputResolution;
    this.outputStride = config.outputStride;
    this.architecture = config.architecture;
  }

  /**
   * Loads the PoseNet model instance from a checkpoint, with the ResNet
   * or MobileNet architecture. The model to be loaded is configurable using the
   * config dictionary ModelConfig. Please find more details in the
   * documentation of the ModelConfig.
   *
   * @param config ModelConfig dictionary that contains parameters for
   * the PoseNet loading process. Please find more details of each parameters
   * in the documentation of the ModelConfig interface. The predefined
   * `MOBILENET_V1_CONFIG` and `RESNET_CONFIG` can also be used as references
   * for defining your customized config.
   */
  static async load(modelConfig: PosenetModelConfig = MOBILENET_V1_CONFIG):
      Promise<PoseDetector> {
    const config = validateModelConfig(modelConfig);
    if (config.architecture === 'ResNet50') {
      // Load ResNet50 model.
      const defaultUrl =
          resNet50Checkpoint(config.outputStride, config.quantBytes);
      const model = await tfconv.loadGraphModel(config.modelUrl || defaultUrl);

      return new PosenetDetector(model, config);
    }

    // Load MobileNetV1 model.
    const defaultUrl = mobileNetCheckpoint(
        config.outputStride, config.multiplier, config.quantBytes);
    const model = await tfconv.loadGraphModel(config.modelUrl || defaultUrl);

    return new PosenetDetector(model, config);
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It returns a
   * single pose or multiple poses based on the maxPose parameter from the
   * `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config
   *       maxPoses: Optional. Max number of poses to estimate.
   *       When maxPoses = 1, a single pose is detected, it is usually much more
   *       efficient than maxPoses > 1. When maxPoses > 1, multiple poses are
   *       detected.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig:
          PoseNetEstimationConfig = SINGLE_PERSON_ESTIMATION_CONFIG):
      Promise<Pose[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      return [];
    }

    this.maxPoses = config.maxPoses;

    const {imageTensor, padding} = convertImageToTensor(
        image, {inputResolution: this.inputResolution, keepAspectRatio: true});

    const imageValueShifted = this.architecture === 'ResNet50' ?
        tf.add(imageTensor, RESNET_MEAN) :
        shiftImageValue(imageTensor, [-1, 1]);

    const results =
        this.posenetModel.predict(imageValueShifted) as tf.Tensor4D[];

    let offsets, heatmap, displacementFwd, displacementBwd;
    if (this.architecture === 'ResNet50') {
      offsets = tf.squeeze(results[2]);
      heatmap = tf.squeeze(results[3]);
      displacementFwd = tf.squeeze(results[0]);
      displacementBwd = tf.squeeze(results[1]);
    } else {
      offsets = tf.squeeze(results[0]);
      heatmap = tf.squeeze(results[1]);
      displacementFwd = tf.squeeze(results[2]);
      displacementBwd = tf.squeeze(results[3]);
    }
    const heatmapScores = tf.sigmoid(heatmap) as tf.Tensor3D;

    let poses;

    if (this.maxPoses === 1) {
      const pose = await decodeSinglePose(
          heatmapScores, offsets as tf.Tensor3D, this.outputStride);
      poses = [pose];
    } else {
      poses = await decodeMultiplePoses(
          heatmapScores, offsets as tf.Tensor3D, displacementFwd as tf.Tensor3D,
          displacementBwd as tf.Tensor3D, this.outputStride, this.maxPoses,
          config.scoreThreshold, config.nmsRadius);
    }

    const imageSize = getImageSize(image);
    let scaledPoses =
        scalePoses(poses, imageSize, this.inputResolution, padding);

    if (config.flipHorizontal) {
      scaledPoses = flipPosesHorizontal(scaledPoses, imageSize);
    }

    imageTensor.dispose();
    imageValueShifted.dispose();
    tf.dispose(results);
    offsets.dispose();
    heatmap.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    heatmapScores.dispose();

    return scaledPoses;
  }

  dispose() {
    this.posenetModel.dispose();
  }

  reset() {
    // No-op. There's no global state.
  }
}
