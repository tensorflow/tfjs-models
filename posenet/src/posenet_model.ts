/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {BaseModel} from './base_model';
import {mobileNetCheckpoint, resNet50Checkpoint} from './checkpoints';
import {MobileNet} from './mobilenet';
import {decodeMultiplePoses} from './multi_pose/decode_multiple_poses';
import {ResNet} from './resnet';
import {decodeSinglePose} from './single_pose/decode_single_pose';
import {InputResolution, MobileNetMultiplier, Pose, PoseNetArchitecture, PosenetInput, PoseNetOutputStride, PoseNetQuantBytes} from './types';
import {assertValidOutputStride, assertValidResolution, getInputTensorDimensions, getValidInputResolutionDimensions, padAndResizeTo, scaleAndFlipPoses, toTensorBuffers3D, validateInputResolution} from './util';

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
 * * `inputResolution`: A number or an Object of type {width: number, height:
 * number}. Specifies the size the input image is scaled to before feeding it
 * through the PoseNet model.  The larger the value, more accurate the model at
 * the cost of speed. Set this to a smaller value to increase speed at the cost
 * of accuracy. If a number is provided, the input will be resized and padded to
 * be a square with the same width and height.  If width and height are
 * provided, the input will be resized and padded to the specified width and
 * height.
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
  inputResolution: InputResolution;
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
//   quantBytes: 2,
// } as ModelConfig;
// ```
const MOBILENET_V1_CONFIG: ModelConfig = {
  architecture: 'MobileNetV1',
  outputStride: 16,
  multiplier: 0.75,
  inputResolution: 257,
} as ModelConfig;

const VALID_ARCHITECTURE = ['MobileNetV1', 'ResNet50'];
const VALID_STRIDE = {
  'MobileNetV1': [8, 16, 32],
  'ResNet50': [32, 16]
};

const VALID_MULTIPLIER = {
  'MobileNetV1': [0.50, 0.75, 1.0],
  'ResNet50': [1.0]
};
const VALID_QUANT_BYTES = [1, 2, 4];

function validateModelConfig(config: ModelConfig) {
  config = config || MOBILENET_V1_CONFIG;

  if (config.architecture == null) {
    config.architecture = 'MobileNetV1';
  }
  if (VALID_ARCHITECTURE.indexOf(config.architecture) < 0) {
    throw new Error(
        `Invalid architecture ${config.architecture}. ` +
        `Should be one of ${VALID_ARCHITECTURE}`);
  }

  if (config.inputResolution == null) {
    config.inputResolution = 257;
  }

  validateInputResolution(config.inputResolution);

  if (config.outputStride == null) {
    config.outputStride = 16;
  }
  if (VALID_STRIDE[config.architecture].indexOf(config.outputStride) < 0) {
    throw new Error(
        `Invalid outputStride ${config.outputStride}. ` +
        `Should be one of ${VALID_STRIDE[config.architecture]} ` +
        `for architecture ${config.architecture}.`);
  }

  if (config.multiplier == null) {
    config.multiplier = 1.0;
  }
  if (VALID_MULTIPLIER[config.architecture].indexOf(config.multiplier) < 0) {
    throw new Error(
        `Invalid multiplier ${config.multiplier}. ` +
        `Should be one of ${VALID_MULTIPLIER[config.architecture]} ` +
        `for architecture ${config.architecture}.`);
  }

  if (config.quantBytes == null) {
    config.quantBytes = 4;
  }
  if (VALID_QUANT_BYTES.indexOf(config.quantBytes) < 0) {
    throw new Error(
        `Invalid quantBytes ${config.quantBytes}. ` +
        `Should be one of ${VALID_QUANT_BYTES} ` +
        `for architecture ${config.architecture}.`);
  }

  if (config.architecture === 'MobileNetV1' && config.outputStride === 32 &&
      config.multiplier !== 1) {
    throw new Error(
        `When using an output stride of 32, ` +
        `you must select 1 as the multiplier.`);
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
 */
export interface InferenceConfig {
  flipHorizontal: boolean;
}

/**
 * Single Person Inference Config
 */
export interface SinglePersonInterfaceConfig extends InferenceConfig {}

/**
 * Multiple Person Inference Config
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
export interface MultiPersonInferenceConfig extends InferenceConfig {
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
}

// these added back to not break the existing api.
export interface LegacyMultiPersonInferenceConfig extends
    MultiPersonInferenceConfig {
  decodingMethod: 'multi-person';
}

export interface LegacySinglePersonInferenceConfig extends
    SinglePersonInterfaceConfig {
  decodingMethod: 'single-person';
}

export const SINGLE_PERSON_INFERENCE_CONFIG: SinglePersonInterfaceConfig = {
  flipHorizontal: false
};

export const MULTI_PERSON_INFERENCE_CONFIG: MultiPersonInferenceConfig = {
  flipHorizontal: false,
  maxDetections: 5,
  scoreThreshold: 0.5,
  nmsRadius: 20
};

function validateSinglePersonInferenceConfig(
    config: SinglePersonInterfaceConfig) {}

function validateMultiPersonInputConfig(config: MultiPersonInferenceConfig) {
  const {maxDetections, scoreThreshold, nmsRadius} = config;

  if (maxDetections <= 0) {
    throw new Error(
        `Invalid maxDetections ${maxDetections}. ` +
        `Should be > 0`);
  }

  if (scoreThreshold < 0.0 || scoreThreshold > 1.0) {
    throw new Error(
        `Invalid scoreThreshold ${scoreThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }

  if (nmsRadius <= 0) {
    throw new Error(`Invalid nmsRadius ${nmsRadius}.`);
  }
}

export class PoseNet {
  readonly baseModel: BaseModel;
  readonly inputResolution: [number, number];

  constructor(net: BaseModel, inputResolution: [number, number]) {
    assertValidOutputStride(net.outputStride);
    assertValidResolution(inputResolution, net.outputStride);

    this.baseModel = net;
    this.inputResolution = inputResolution;
  }

  /**
   * Infer through PoseNet, and estimates multiple poses using the outputs.
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It detects
   * multiple poses and finds their parts from part scores and displacement
   * vectors using a fast greedy decoding algorithm.  It returns up to
   * `config.maxDetections` object instance detections in decreasing root
   * score order.
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
   * image to feed through the network.
   *
   * @param config MultiPoseEstimationConfig object that contains parameters
   * for the PoseNet inference using multiple pose estimation.
   *
   * @return An array of poses and their scores, each containing keypoints and
   * the corresponding keypoint scores.  The positions of the keypoints are
   * in the same scale as the original image
   */
  async estimateMultiplePoses(
      input: PosenetInput,
      config: MultiPersonInferenceConfig = MULTI_PERSON_INFERENCE_CONFIG):
      Promise<Pose[]> {
    const configWithDefaults: MultiPersonInferenceConfig = {
      ...MULTI_PERSON_INFERENCE_CONFIG,
      ...config
    };

    validateMultiPersonInputConfig(config);

    const outputStride = this.baseModel.outputStride;
    const inputResolution = this.inputResolution;

    const [height, width] = getInputTensorDimensions(input);

    const {resized, padding} = padAndResizeTo(input, inputResolution);

    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        this.baseModel.predict(resized);

    const allTensorBuffers = await toTensorBuffers3D(
        [heatmapScores, offsets, displacementFwd, displacementBwd]);

    const scoresBuffer = allTensorBuffers[0];
    const offsetsBuffer = allTensorBuffers[1];
    const displacementsFwdBuffer = allTensorBuffers[2];
    const displacementsBwdBuffer = allTensorBuffers[3];

    const poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, outputStride, configWithDefaults.maxDetections,
        configWithDefaults.scoreThreshold, configWithDefaults.nmsRadius);

    const resultPoses = scaleAndFlipPoses(
        poses, [height, width], inputResolution, padding,
        configWithDefaults.flipHorizontal);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    resized.dispose();

    return resultPoses;
  }

  /**
   * Infer through PoseNet, and estimates a single pose using the outputs.
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It detects
   * multiple poses and finds their parts from part scores and displacement
   * vectors using a fast greedy decoding algorithm.  It returns a single pose
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
   * image to feed through the network.
   *
   * @param config SinglePersonEstimationConfig object that contains
   * parameters for the PoseNet inference using single pose estimation.
   *
   * @return An pose and its scores, containing keypoints and
   * the corresponding keypoint scores.  The positions of the keypoints are
   * in the same scale as the original image
   */
  async estimateSinglePose(
      input: PosenetInput,
      config: SinglePersonInterfaceConfig = SINGLE_PERSON_INFERENCE_CONFIG):
      Promise<Pose> {
    const configWithDefaults = {...SINGLE_PERSON_INFERENCE_CONFIG, ...config};

    validateSinglePersonInferenceConfig(configWithDefaults);

    const outputStride = this.baseModel.outputStride;
    const inputResolution = this.inputResolution;

    const [height, width] = getInputTensorDimensions(input);

    const {resized, padding} = padAndResizeTo(input, inputResolution);

    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        this.baseModel.predict(resized);

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);
    const poses = [pose];

    const resultPoses = scaleAndFlipPoses(
        poses, [height, width], inputResolution, padding,
        configWithDefaults.flipHorizontal);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    resized.dispose();

    return resultPoses[0];
  }

  /** Deprecated: Use either estimateSinglePose or estimateMultiplePoses */
  async estimatePoses(
      input: PosenetInput,
      config: LegacySinglePersonInferenceConfig|
      LegacyMultiPersonInferenceConfig): Promise<Pose[]> {
    if (config.decodingMethod === 'single-person') {
      const pose = await this.estimateSinglePose(input, config);
      return [pose];
    } else {
      return this.estimateMultiplePoses(input, config);
    }
  }

  public dispose() {
    this.baseModel.dispose();
  }
}

async function loadMobileNet(config: ModelConfig): Promise<PoseNet> {
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  const multiplier = config.multiplier;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  const url = mobileNetCheckpoint(outputStride, multiplier, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const mobilenet = new MobileNet(graphModel, outputStride);

  const validInputResolution = getValidInputResolutionDimensions(
      config.inputResolution, mobilenet.outputStride);

  return new PoseNet(mobilenet, validInputResolution);
}

async function loadResNet(config: ModelConfig): Promise<PoseNet> {
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
  const resnet = new ResNet(graphModel, outputStride);
  const validInputResolution = getValidInputResolutionDimensions(
      config.inputResolution, resnet.outputStride);
  return new PoseNet(resnet, validInputResolution);
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
