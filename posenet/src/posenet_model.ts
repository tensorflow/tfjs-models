/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {mobileNetCheckpoint, resNet50Checkpoint} from './checkpoints';
import {assertValidOutputStride, assertValidResolution, MobileNet, MobileNetMultiplier} from './mobilenet';
import {decodeMultiplePoses} from './multi_pose/decode_multiple_poses';
import {ResNet} from './resnet';
import {decodeSinglePose} from './single_pose/decode_single_pose';
import {Pose, PosenetInput} from './types';
import {flipPosesHorizontal, getInputTensorDimensions, padAndResizeTo, scalePoses, toTensorBuffers3D} from './util';

export type PoseNetResolution =
    161|193|257|289|321|353|385|417|449|481|513|801|1217;
export type PoseNetOutputStride = 32|16|8;
export type PoseNetArchitecture = 'ResNet50'|'MobileNetV1';
export type PoseNetDecodingMethod = 'single-person'|'multi-person';
export type PoseNetQuantBytes = 1|2|4;

/**
 * PoseNet supports using various convolution neural network models
 * (e.g. ResNet and MobileNetV1) as its underlying base model.
 * The following BaseModel interface defines a unified interface for
 * creating such PoseNet base models. Currently both MobileNet (in
 * ./mobilenet.ts) and ResNet (in ./resnet.ts) implements the BaseModel
 * interface. New base models that conform to the BaseModel interface can be
 * added to PoseNet.
 */
export interface BaseModel {
  // The input resolution (unit: pixel) of the base model.
  readonly inputResolution: PoseNetResolution;
  // The output stride of the base model.
  readonly outputStride: PoseNetOutputStride;

  /**
   * Predicts intermediate Tensor representations.
   *
   * @param input The input RGB image of the base model.
   * A Tensor of shape: [`inputResolution`, `inputResolution`, 3].
   *
   * @return A dictionary of base model's intermediate predictions.
   * The returned dictionary should contains the following elements:
   * heatmapScores: A Tensor3D that represents the heatmapScores.
   * offsets: A Tensor3D that represents the offsets.
   * displacementFwd: A Tensor3D that represents the forward displacement.
   * displacementBwd: A Tensor3D that represents the backward displacement.
   */
  predict(input: tf.Tensor3D): {[key: string]: tf.Tensor3D};
  /**
   * Releases the CPU and GPU memory allocated by the model.
   */
  dispose(): void;
}

/**
 * PoseNet model loading is configurable using the following config dictionary.
 *
 * `architecture`: PoseNetArchitecture. It determines wich PoseNet architecture
 * to load. The supported architectures are: MobileNetV1 and ResNet.
 *
 * `outputStride`: Specifies the output stride of the PoseNet model.
 * The smaller the value, the larger the output resolution, and more accurate
 * the model at the cost of speed.  Set this to a larger value to increase speed
 * at the cost of accuracy. Stride 32 is supported for ResNet and
 * stride 8,16,32 are supported for various MobileNetV1 models.
 *
 * `inputResolution`: Specifies the input resolution of the PoseNet model.
 * The larger the value, more accurate the model at the cost of speed.
 * Set this to a smaller value to increase speed at the cost of accuracy.
 * Input resolution 257 and 513 are supported for ResNet and any resolution in
 * PoseNetResolution are supported for MobileNetV1.
 *
 * `multiplier`: An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. The value is used only by MobileNet architecture. It is the float
 * multiplier for the depth (number of channels) for all convolution ops.
 * The larger the value, the larger the size of the layers, and more accurate
 * the model at the cost of speed. Set this to a smaller value to increase speed
 * at the cost of accuracy.
 *
 * `modelUrl`: An optional string that specifies custom url of the model. This
 * is useful for area/countries that don't have access to the model hosted on
 * GCP.
 *
 * `quantBytes`: An opional number with values: 1, 2, or 4.  This parameter
 * affects weight quantization in the models. The available options are
 * 1 byte, 2 bytes, and 4 bytes. The higher the value, the larger the model size
 * and thus the longer the loading time, the lower the value, the shorter the
 * loading time but lower the accuracy.
 */
export interface ModelConfig {
  architecture: PoseNetArchitecture;
  outputStride: PoseNetOutputStride;
  inputResolution: PoseNetResolution;
  multiplier?: MobileNetMultiplier;
  modelUrl?: string;
  quantBytes?: PoseNetQuantBytes;
}

// The default configuration for loading MobileNetV1 based PoseNet.
//
// (And for references, the default configuration for loading ResNet
// based PoseNet is also included).
//
// ```
// const RESNET_CONFIG = {
//   architecture: 'ResNet50',
//   outputStride: 32,
//   inputResolution: 257,
//   quantBytes: 2,
// } as ModelConfig;
// ```
const MOBILENET_V1_CONFIG: ModelConfig = {
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: 513,
  multiplier: 0.75,
} as ModelConfig;

function validateModelConfig(config: ModelConfig) {
  config = config || MOBILENET_V1_CONFIG;

  const VALID_ARCHITECTURE = ['MobileNetV1', 'ResNet50'];
  const VALID_STRIDE = {'MobileNetV1': [8, 16, 32], 'ResNet50': [32, 16]};
  const VALID_RESOLUTION =
      [161, 193, 257, 289, 321, 353, 385, 417, 449, 481, 513, 801];
  const VALID_MULTIPLIER = {
    'MobileNetV1': [0.50, 0.75, 1.0],
    'ResNet50': [1.0]
  };
  const VALID_QUANT_BYTES = [1, 2, 4];

  if (config.architecture == null) {
    config.architecture = 'MobileNetV1';
  }
  if (VALID_ARCHITECTURE.indexOf(config.architecture) < 0) {
    throw new Error(
        `Invalid architecture ${config.architecture}. ` +
        `Should be one of ${VALID_ARCHITECTURE}`);
  }

  if (config.outputStride == null) {
    config.outputStride = 16;
  }
  if (VALID_STRIDE[config.architecture].indexOf(config.outputStride) < 0) {
    throw new Error(
        `Invalid outputStride ${config.outputStride}. ` +
        `Should be one of ${VALID_STRIDE[config.architecture]} ` +
        `for architecutre ${config.architecture}.`);
  }

  if (config.inputResolution == null) {
    config.inputResolution = 513;
  }
  if (VALID_RESOLUTION.indexOf(config.inputResolution) < 0) {
    throw new Error(
        `Invalid inputResolution ${config.inputResolution}. ` +
        `Should be one of ${VALID_RESOLUTION} ` +
        `for architecutre ${config.architecture}.`);
  }

  if (config.multiplier == null) {
    config.multiplier = 1.0;
  }
  if (VALID_MULTIPLIER[config.architecture].indexOf(config.multiplier) < 0) {
    throw new Error(
        `Invalid multiplier ${config.multiplier}. ` +
        `Should be one of ${VALID_MULTIPLIER[config.architecture]} ` +
        `for architecutre ${config.architecture}.`);
  }

  if (config.quantBytes == null) {
    config.quantBytes = 4;
  }
  if (VALID_QUANT_BYTES.indexOf(config.quantBytes) < 0) {
    throw new Error(
        `Invalid quantBytes ${config.quantBytes}. ` +
        `Should be one of ${VALID_QUANT_BYTES} ` +
        `for architecutre ${config.architecture}.`);
  }

  return config;
}

/**
 * PoseNet inference is configurable using the following config dictionary.
 *
 * `flipHorizontal`: If the poses should be flipped/mirrored horizontally.
 * This should be set to true for videos where the video is by default flipped
 * horizontally (i.e. a webcam), and you want the poses to be returned in the
 * proper orientation.
 *
 * `decodingMethod`: PoseNetDecodingMethod. Whether to use "single-person" or
 * "multi-person" PoseNet decoding algorithm to decode the predicted
 * intermediate tensors. "single-person" decoding works only when there is one
 * person in the input image and "multi-person" decoding works even when there
 * are many people in the input image.
 *
 * `maxDetections`: Maximum number of returned instance detections per image.
 *
 * `scoreThreshold`: Only return instance detections that have root part
 * score greater or equal to this value. Defaults to 0.5
 *
 * `nmsRadius`: Non-maximum suppression part distance in pixels. It needs
 * to be strictly positive. Two parts suppress each other if they are less
 * than `nmsRadius` pixels away. Defaults to 20.
 */
export interface InferenceConfig {
  flipHorizontal: boolean;
  decodingMethod: PoseNetDecodingMethod;
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
}

export const SINGLE_PERSON_INFERENCE_CONFIG = {
  flipHorizontal: false,
  decodingMethod: 'single-person',
} as InferenceConfig;

export const MULTI_PERSON_INFERENCE_CONFIG = {
  flipHorizontal: false,
  decodingMethod: 'multi-person',
  maxDetections: 5,
  scoreThreshold: 0.5,
  nmsRadius: 20
} as InferenceConfig;

function validateInferenceConfig(config: InferenceConfig) {
  config = config || MULTI_PERSON_INFERENCE_CONFIG;
  const VALID_DECODING_METHOD = ['single-person', 'multi-person'];

  if (config.flipHorizontal == null) {
    config.flipHorizontal = false;
  }

  if (config.decodingMethod == null) {
    config.decodingMethod = 'multi-person';
  }
  if (VALID_DECODING_METHOD.indexOf(config.decodingMethod) < 0) {
    throw new Error(
        `Invalid decoding method ${config.decodingMethod}. ` +
        `Should be one of ${VALID_DECODING_METHOD}`);
  }

  // Validates parameters used only by multi-person decoding.
  if (config.decodingMethod === 'multi-person') {
    if (config.maxDetections == null) {
      config.maxDetections = 5;
    }
    if (config.maxDetections <= 0) {
      throw new Error(
          `Invalid maxDetections ${config.maxDetections}. ` +
          `Should be > 0 for decodingMethod ${config.decodingMethod}.`);
    }

    if (config.scoreThreshold == null) {
      config.scoreThreshold = 0.5;
    }
    if (config.scoreThreshold < 0.0 || config.scoreThreshold > 1.0) {
      throw new Error(
          `Invalid scoreThreshold ${config.scoreThreshold}. ` +
          `Should be in range [0.0, 1.0] for decodingMethod ${
              config.decodingMethod}.`);
    }

    if (config.nmsRadius == null) {
      config.nmsRadius = 20;
    }
    if (config.nmsRadius <= 0) {
      throw new Error(
          `Invalid nmsRadius ${config.nmsRadius}. ` +
          `Should be positive for decodingMethod ${config.decodingMethod}.`);
    }
  }
  return config;
}

export class PoseNet {
  baseModel: BaseModel;

  constructor(net: BaseModel) {
    this.baseModel = net;
  }

  /**
   * Infer through PoseNet, and estimates multiple poses using the outputs.
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It detects
   * multiple poses and finds their parts from part scores and displacement
   * vectors using a fast greedy decoding algorithm.  It returns up to
   * `config.maxDetections` object instance detections in decreasing root score
   *  order.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config InferenceConfig dictionary that contains parameters for
   * the PoseNet inference process. Please find more details of each parameters
   * in the documentation of the InferenceConfig interface. The predefined
   * `MULTI_PERSON_INFERENCE_CONFIG` and `SINGLE_PERSON_INFERENCE_CONFIG` can
   * also be used as references for defining your customized config.
   *
   * @return An array of poses and their scores, each containing keypoints and
   * the corresponding keypoint scores.  The positions of the keypoints are
   * in the same scale as the original image
   */
  async estimatePoses(
      input: PosenetInput,
      config: InferenceConfig = MULTI_PERSON_INFERENCE_CONFIG):
      Promise<Pose[]> {
    config = validateInferenceConfig(config);
    const outputStride = this.baseModel.outputStride;
    const inputResolution = this.baseModel.inputResolution;
    assertValidOutputStride(outputStride);
    assertValidResolution(this.baseModel.inputResolution, outputStride);

    const [height, width] = getInputTensorDimensions(input);
    let [resizedHeight, resizedWidth] = [0, 0];
    let [padTop, padBottom, padLeft, padRight] = [0, 0, 0, 0];
    let heatmapScores, offsets, displacementFwd, displacementBwd;

    resizedHeight = inputResolution;
    resizedWidth = inputResolution;

    const outputs = tf.tidy(() => {
      const {resized, paddedBy} =
          padAndResizeTo(input, [resizedHeight, resizedWidth]);
      padTop = paddedBy[0][0];
      padBottom = paddedBy[0][1];
      padLeft = paddedBy[1][0];
      padRight = paddedBy[1][1];
      return this.baseModel.predict(resized);
    });
    heatmapScores = outputs.heatmapScores;
    offsets = outputs.offsets;
    displacementFwd = outputs.displacementFwd;
    displacementBwd = outputs.displacementBwd;

    let poses;
    if (config.decodingMethod === 'multi-person') {
      const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
          await toTensorBuffers3D(
              [heatmapScores, offsets, displacementFwd, displacementBwd]);

      poses = await decodeMultiplePoses(
          scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
          displacementsBwdBuffer, outputStride, config.maxDetections,
          config.scoreThreshold, config.nmsRadius);
    } else {
      const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);
      poses = [pose];
    }

    const scaleY = (height + padTop + padBottom) / (resizedHeight);
    const scaleX = (width + padLeft + padRight) / (resizedWidth);
    let scaledPoses = scalePoses(poses, scaleY, scaleX, -padTop, -padLeft);

    if (config.flipHorizontal) {
      scaledPoses = flipPosesHorizontal(scaledPoses, width);
    }

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return scaledPoses;
  }

  public dispose() {
    this.baseModel.dispose();
  }
}

async function loadMobileNet(config: ModelConfig): Promise<PoseNet> {
  const inputResolution = config.inputResolution;
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  const multiplier = config.multiplier;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  const url = mobileNetCheckpoint(
      inputResolution, outputStride, multiplier, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const mobilenet = new MobileNet(graphModel, inputResolution, outputStride);
  return new PoseNet(mobilenet);
}

async function loadResNet(config: ModelConfig): Promise<PoseNet> {
  const inputResolution = config.inputResolution;
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  const url = resNet50Checkpoint(outputStride, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const resnet = new ResNet(graphModel, inputResolution, outputStride);
  return new PoseNet(resnet);
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
export async function load(config: ModelConfig = MOBILENET_V1_CONFIG):
    Promise<PoseNet> {
  config = validateModelConfig(config);
  if (config.architecture === 'ResNet50') {
    return loadResNet(config);
  } else if (config.architecture === 'MobileNetV1') {
    return loadMobileNet(config);
  } else {
    return null;
  }
}
