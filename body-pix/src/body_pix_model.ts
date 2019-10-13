
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

import {decodeOnlyPartSegmentation, decodePartSegmentation, toMaskTensor} from './decode_part_map';
import {MobileNet, MobileNetMultiplier} from './mobilenet';
import {decodeMultipleMasksGPU, decodeMultiplePartMasksGPU} from './multi_person/decode_multiple_masks';
import {decodeMultiplePoses} from './multi_person/decode_multiple_poses';
import {ResNet} from './resnet';
import {mobileNetSavedModel, resNet50SavedModel} from './saved_models';
import {decodeSinglePose} from './sinlge_person/decode_single_pose';
import {BodyPixInput, InputResolution, Padding, PartSegmentation, PersonSegmentation} from './types';
import {assertValidOutputStride, assertValidResolution, getInputTensorDimensions, getValidInputResolutionDimensions, padAndResizeTo, scaleAndCropToInputTensorShape, scaleAndFlipPoses, toInputTensor, toTensorBuffers3D, validateInputResolution} from './util';

export type BodyPixOutputStride = 32|16|8;
export type BodyPixArchitecture = 'ResNet50'|'MobileNetV1';
export type BodyPixDecodingMethod = 'person'|'multi-person-by-instance';
export type BodyPixQuantBytes = 1|2|4;
export type BodyPixMultiplier = 1.0|0.75|0.50;

const APPLY_SIGMOID_ACTIVATION = true;

/**
 * BodyPix supports using various convolution neural network models
 * (e.g. ResNet and MobileNetV1) as its underlying base model.
 * The following BaseModel interface defines a unified interface for
 * creating such BodyPix base models. Currently both MobileNet (in
 * ./mobilenet.ts) and ResNet (in ./resnet.ts) implements the BaseModel
 * interface. New base models that conform to the BaseModel interface can be
 * added to BodyPix.
 */
export interface BaseModel {
  // The output stride of the base model.
  readonly outputStride: BodyPixOutputStride;

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
   * - segmentation: A Tensor3D that represents the segmentation of all people.
   * - longOffsets: A Tensor3D that represents the long offsets used for
   * instance grouping.
   * - partHeatmaps: A Tensor3D that represents the body part segmentation.
   */
  predict(input: tf.Tensor3D): {[key: string]: tf.Tensor3D};
  /**
   * Releases the CPU and GPU memory allocated by the model.
   */
  dispose(): void;
}

/**
 * BodyPix model loading is configurable using the following config dictionary.
 *
 * `architecture`: BodyPixArchitecture. It determines which BodyPix architecture
 * to load. The supported architectures are: MobileNetV1 and ResNet50.
 *
 * `outputStride`: Specifies the output stride of the BodyPix model.
 * The smaller the value, the larger the output resolution, and more accurate
 * the model at the cost of speed. Set this to a larger value to increase speed
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
 * height
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
 * `quantBytes`: An optional number with values: 1, 2, or 4.  This parameter
 * affects weight quantization in the models. The available options are
 * 1 byte, 2 bytes, and 4 bytes. The higher the value, the larger the model size
 * and thus the longer the loading time, the lower the value, the shorter the
 * loading time but lower the accuracy.
 */
export interface ModelConfig {
  architecture: BodyPixArchitecture;
  outputStride: BodyPixOutputStride;
  inputResolution: InputResolution;
  multiplier?: MobileNetMultiplier;
  modelUrl?: string;
  quantBytes?: BodyPixQuantBytes;
}

// The default configuration for loading MobileNetV1 based BodyPix.
//
// (And for references, the default configuration for loading ResNet
// based PoseNet is also included).
//
// ```
// const RESNET_CONFIG = {
//   architecture: 'ResNet50',
//   outputStride: 32,
//   inputResolution: 513,
//   quantBytes: 4,
// } as ModelConfig;
// ```

const MOBILENET_V1_CONFIG = {
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: 513,
  quantBytes: 4,
  multiplier: 0.75,
} as ModelConfig;

const VALID_ARCHITECTURE: BodyPixArchitecture[] = ['MobileNetV1', 'ResNet50'];
const VALID_STRIDE: {[id: string]: BodyPixOutputStride[]} = {
  'MobileNetV1': [8, 16, 32],
  'ResNet50': [32, 16]
};
const VALID_MULTIPLIER: {[id: string]: BodyPixMultiplier[]} = {
  'MobileNetV1': [0.50, 0.75, 1.0],
  'ResNet50': [1.0]
};
const VALID_QUANT_BYTES: BodyPixQuantBytes[] = [1, 2, 4];

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

  return config;
}

/**
 * BodyPix inference is configurable using the following config dictionary.
 *
 * `flipHorizontal`: If the left-right keypoint of poses/part segmentation
 * should be flipped/mirrored horizontally. This should be set to true for
 * videos where the video is by default flipped horizontally (i.e. a webcam),
 * and you want the person & body part segmentation to be returned in the proper
 * orientation.
 *
 *  `segmentationThreshold`: The minimum that segmentation values must
 * have to be considered part of the person. Affects the generation of the
 * segmentation mask. More specifically, it is the threshold used to binarize
 * the intermediate person segmentation probability. The probability of each
 * pixel belongs to a person is in range [0, 1]. If the probability is greater
 * than the `segmentationThreshold`, it will be set to 1 otherwise 0.
 *
 */
export interface InferenceConfig {
  flipHorizontal: boolean;
  segmentationThreshold: number;
}

/**
 * Person Inference Config
 */
export interface PersonInferenceConfig extends InferenceConfig {}

/**
 * Multiple Person Instance Inference Config
 *
 * `maxDetections`: Maximum number of returned instance detections per image.
 * Defaults to 10
 *
 * `scoreThreshold`: Only return instance detections that have root part
 * score greater or equal to this value. Defaults to 0.7
 *
 * `nmsRadius`: Non-maximum suppression part distance in pixels. It needs
 * to be strictly positive. Two parts suppress each other if they are less
 * than `nmsRadius` pixels away. Defaults to 20.
 *
 * `minKeypointScore`: Default to 0.3. Keypoints above the score are used
 * for matching and assigning segmentation mask to each person..
 *
 * `refineSteps`: The number of refinement steps used when assigning the
 * instance segmentation. It needs to be strictly positive. The larger the
 * higher the accuracy and slower the inference.
 *
 */
export interface MultiPersonInstanceInferenceConfig extends InferenceConfig {
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
  minKeypointScore?: number;
  refineSteps?: number;
}

export const PERSON_INFERENCE_CONFIG: PersonInferenceConfig = {
  flipHorizontal: false,
  segmentationThreshold: 0.7
};

export const MULTI_PERSON_INSTANCE_INFERENCE_CONFIG:
    MultiPersonInstanceInferenceConfig = {
      flipHorizontal: false,
      segmentationThreshold: 0.7,
      maxDetections: 10,
      scoreThreshold: 0.2,
      nmsRadius: 20,
      minKeypointScore: 0.3,
      refineSteps: 10
    };

function validatePersonInferenceConfig(config: PersonInferenceConfig) {
  const segmentationThreshold = config.segmentationThreshold;
  if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
    throw new Error(
        `segmentationThreshold ${segmentationThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }
}

function validateMultiPersonInstanceInferenceConfig(
    config: MultiPersonInstanceInferenceConfig) {
  const {
    maxDetections,
    scoreThreshold,
    nmsRadius,
    minKeypointScore,
    refineSteps
  } = config;

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

  if (minKeypointScore < 0 || minKeypointScore > 1) {
    throw new Error(
        `Invalid minKeypointScore ${minKeypointScore}.` +
        `Should be in range [0.0, 1.0]`);
  }

  if (refineSteps <= 0 || minKeypointScore > 20) {
    throw new Error(
        `Invalid refineSteps ${refineSteps}.` +
        `Should be in range [1, 20]`);
  }
}

export class BodyPix {
  readonly baseModel: BaseModel;
  readonly inputResolution: [number, number];

  constructor(net: BaseModel, inputResolution: [number, number]) {
    assertValidOutputStride(net.outputStride);
    assertValidResolution(inputResolution, net.outputStride);

    this.baseModel = net;
    this.inputResolution = inputResolution;
  }

  private predictForPersonSegmentation(input: tf.Tensor3D): {
    segmentLogits: tf.Tensor3D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
  } {
    const {
      segmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
    } = this.baseModel.predict(input);
    return {
      segmentLogits: segmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
    };
  }

  private predictForPersonSegmentationAndPart(input: tf.Tensor3D): {
    segmentLogits: tf.Tensor3D,
    partHeatmapLogits: tf.Tensor3D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
  } {
    const {
      segmentation,
      partHeatmaps,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd
    } = this.baseModel.predict(input);
    return {
      segmentLogits: segmentation,
      partHeatmapLogits: partHeatmaps,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
    };
  }

  private predictForMultiPersonInstanceSegmentationAndPart(input: tf.Tensor3D):
      {
        segmentLogits: tf.Tensor3D,
        longOffsets: tf.Tensor3D,
        heatmapScores: tf.Tensor3D,
        offsets: tf.Tensor3D,
        displacementFwd: tf.Tensor3D,
        displacementBwd: tf.Tensor3D,
        partHeatmaps: tf.Tensor3D
      } {
    const {
      segmentation,
      longOffsets,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      partHeatmaps,
    } = this.baseModel.predict(input);
    return {
      segmentLogits: segmentation,
      longOffsets,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      partHeatmaps
    };
  }

  /**
   * Given an image with people, returns a dictionary of all intermediate
   * tensors including: 1) a binary array with 1 for the pixels that are part of
   * the person, and 0 otherwise, 2) heatmapScores, 3) offsets, and 4) paddings.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person. Affects the generation of the
   * segmentation mask.
   *
   * @return A dictionary containing `segmentation`, `heatmapScores`, `offsets`,
   * and `padding`:
   * - `segmentation`: A 2d Tensor with 1 for the pixels that are part of the
   * person, and 0 otherwise. The width and height correspond to the same
   * dimensions of the input image.
   * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
   * single-person pose estimation decoding.
   * - `offsets`: A 3d Tensor of the keypoint offsets used by single-person pose
   * estimation decoding.
   * - `padding`: The padding (unit pixels) being applied to the input image
   * before it is fed into the model.
   */
  segmentPersonActivation(input: BodyPixInput, segmentationThreshold = 0.5): {
    segmentation: tf.Tensor2D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    padding: Padding
  } {
    const inputResolution = this.inputResolution;
    const imageTensor = toInputTensor(input);
    const {resized, padding} = padAndResizeTo(imageTensor, inputResolution);

    const {segmentation, heatmapScores, offsets} = tf.tidy(() => {
      const {
        segmentLogits,
        heatmapScores,
        offsets,
      } = this.predictForPersonSegmentation(resized);

      const [resizedHeight, resizedWidth] = resized.shape;
      const [height, width] = imageTensor.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      return {
        segmentation:
            toMaskTensor(scaledSegmentScores.squeeze(), segmentationThreshold),
        heatmapScores,
        offsets,
      };
    });
    resized.dispose();
    return {segmentation, heatmapScores, offsets, padding};
  }

  /**
   * Given an image with many people, returns a PersonSegmentation dictionary
   * that contains the segmentation mask for all people and a single pose.
   *
   * Note: The segmentation mask returned by this method covers all people but
   * the pose works well for one person. If you want to estimate instance-level
   * multiple person segmentation & pose for each person, use
   * `segmentMultiPerson` instead.
   *
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config PersonInferenceConfig object that contains
   * parameters for the BodyPix inference using person decoding.
   *
   * @return A PersonSegmentation dictionary that contains height, width, the
   * flattened binary segmentation mask and the pose for the person. The width
   * and height correspond to the same dimensions of the input image.
   * - `height`: The height of the segmentation data in pixel unit.
   * - `width`: The width of the segmentation data in pixel unit.
   * - `data`: The flattened Uint8Array of segmentation data. 1 means the pixel
   * belongs to a person and 0 means the pixel doesn't belong to a person. The
   * size of the array is equal to `height` x `width` in row-major order.
   * - `pose`: The 2d pose of the person.
   */
  async segmentPerson(
      input: BodyPixInput,
      config: PersonInferenceConfig = PERSON_INFERENCE_CONFIG):
      Promise<PersonSegmentation> {
    const configWithDefault:
        PersonInferenceConfig = {...PERSON_INFERENCE_CONFIG, ...config};
    validatePersonInferenceConfig(configWithDefault);
    const {segmentation, heatmapScores, offsets, padding} =
        this.segmentPersonActivation(
            input, configWithDefault.segmentationThreshold);

    const [height, width] = segmentation.shape;

    const result = await segmentation.data() as Uint8Array;
    segmentation.dispose();

    const pose = await decodeSinglePose(
        heatmapScores, offsets, this.baseModel.outputStride);

    const resultPose = scaleAndFlipPoses(
        [pose], [height, width], this.inputResolution, padding,
        configWithDefault.flipHorizontal)[0];

    heatmapScores.dispose();
    offsets.dispose();

    return {height, width, data: result, pose: resultPose};
  }

  /**
   * Given an image with multiple people, returns an *array* of
   * PersonSegmentation object. Each element in the array corresponding to one
   * of the people in the input image. In other words, it predicts
   * instance-level multiple person segmentation & pose for each person.
   *
   * The model does standard ImageNet pre-processing before inferring through
   * the model. The image pixels should have values [0-255].
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
   * image to feed through the network.
   *
   * @param config MultiPersonInferenceConfig object that contains
   * parameters for the BodyPix inference using multi-person decoding.
   *
   * @return An array of PersonSegmentation object, each containing a width,
   * height, a binary array (1 for the pixels that are part of the
   * person, and 0 otherwise) and 2D pose. The array size corresponds to the
   * number of pixels in the image. The width and height correspond to the
   * dimensions of the image the binary array is shaped to, which are the same
   * dimensions of the input image.
   */
  async segmentMultiPerson(
      input: BodyPixInput,
      config: MultiPersonInstanceInferenceConfig =
          MULTI_PERSON_INSTANCE_INFERENCE_CONFIG):
      Promise<PersonSegmentation[]> {
    const configWithDefault: MultiPersonInstanceInferenceConfig = {
      ...MULTI_PERSON_INSTANCE_INFERENCE_CONFIG,
      ...config
    };
    validateMultiPersonInstanceInferenceConfig(configWithDefault);
    const [height, width] = getInputTensorDimensions(input);
    const inputResolution = this.inputResolution;

    const {resized, padding} = padAndResizeTo(input, inputResolution);
    const {
      segmentation,
      longOffsets,
      heatmapScoresRaw,
      offsetsRaw,
      displacementFwdRaw,
      displacementBwdRaw,
    } = tf.tidy(() => {
      const {
        segmentLogits,
        longOffsets,
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd,
      } = this.predictForMultiPersonInstanceSegmentationAndPart(resized);
      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], inputResolution,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);
      const longOffsetsResized = false;
      let scaledLongOffsets;
      if (longOffsetsResized) {
        scaledLongOffsets = scaleAndCropToInputTensorShape(
            longOffsets, [height, width], inputResolution,
            [[padding.top, padding.bottom], [padding.left, padding.right]],
            APPLY_SIGMOID_ACTIVATION);
      } else {
        scaledLongOffsets = longOffsets;
      }

      const segmentation = toMaskTensor(
          scaledSegmentScores.squeeze(),
          configWithDefault.segmentationThreshold);

      return {
        segmentation,
        longOffsets: scaledLongOffsets,
        heatmapScoresRaw: heatmapScores,
        offsetsRaw: offsets,
        displacementFwdRaw: displacementFwd,
        displacementBwdRaw: displacementBwd,
      };
    });

    const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
        await toTensorBuffers3D([
          heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw
        ]);

    let poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, this.baseModel.outputStride,
        config.maxDetections, configWithDefault.scoreThreshold,
        configWithDefault.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], inputResolution, padding, false);

    const instanceMasks = await decodeMultipleMasksGPU(
        segmentation, longOffsets, poses, height, width,
        this.baseModel.outputStride, inputResolution,
        [[padding.top, padding.bottom], [padding.left, padding.right]],
        configWithDefault.scoreThreshold, configWithDefault.refineSteps,
        configWithDefault.minKeypointScore, configWithDefault.maxDetections);

    resized.dispose();
    segmentation.dispose();
    longOffsets.dispose();
    heatmapScoresRaw.dispose();
    offsetsRaw.dispose();
    displacementFwdRaw.dispose();
    displacementBwdRaw.dispose();

    return instanceMasks;
  }

  /**
   * Given an image with many people, returns a dictionary containing: height,
   * width, a tensor with a part id from 0-24 for the pixels that are
   * part of a corresponding body part, and -1 otherwise. This does standard
   * ImageNet pre-processing before inferring through the model.  The image
   * should pixels should have values [0-255].
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the clipping of the colored
   * part image.
   *
   * @return  A dictionary containing `partSegmentation`, `heatmapScores`,
   * `offsets`, and `padding`:
   * - `partSegmentation`: A 2d Tensor with a part id from 0-24 for
   * the pixels that are part of a corresponding body part, and -1 otherwise.
   * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
   * single-person pose estimation decoding.
   * - `offsets`: A 3d Tensor of the keypoint offsets used by single-person pose
   * estimation decoding.
   * - `padding`: The padding (unit pixels) being applied to the input image
   * before it is fed into the model.
   */
  segmentPersonPartsActivation(
      input: BodyPixInput, segmentationThreshold = 0.5): {
    partSegmentation: tf.Tensor2D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    padding: Padding
  } {
    const inputResolution = this.inputResolution;
    const imageTensor = toInputTensor(input);
    const {
      resized,
      padding,
    } = padAndResizeTo(imageTensor, inputResolution);

    const {partSegmentation, heatmapScores, offsets} = tf.tidy(() => {
      const {segmentLogits, partHeatmapLogits, heatmapScores, offsets} =
          this.predictForPersonSegmentationAndPart(resized);

      const [resizedHeight, resizedWidth] = resized.shape;
      const [height, width] = imageTensor.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      const scaledPartHeatmapScore = scaleAndCropToInputTensorShape(
          partHeatmapLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);
      const segmentation =
          toMaskTensor(scaledSegmentScores.squeeze(), segmentationThreshold);
      return {
        partSegmentation:
            decodePartSegmentation(segmentation, scaledPartHeatmapScore),
        heatmapScores,
        offsets
      };
    });
    resized.dispose();
    return {partSegmentation, heatmapScores, offsets, padding};
  }

  /**
   * Given an image with many people, returns a PartSegmentation dictionary that
   * contains the body part segmentation mask for all people and a single pose.
   *
   * Note: The body part segmentation mask returned by this method covers all
   * people but the pose works well when there is one person. If you want to
   * estimate instance-level multiple person body part segmentation & pose for
   * each person, use `segmentMultiPersonParts` instead.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config PersonInferenceConfig object that contains
   * parameters for the BodyPix inference using single person decoding.
   *
   * @return A PersonSegmentation dictionary that contains height, width, the
   * flattened binary segmentation mask and the pose for the person. The width
   * and height correspond to the same dimensions of the input image.
   * - `height`: The height of the person part segmentation data in pixel unit.
   * - `width`: The width of the person part segmentation data in pixel unit.
   * - `data`: The flattened Int32Array of person part segmentation data with a
   * part id from 0-24 for the pixels that are part of a corresponding body
   * part, and -1 otherwise. The size of the array is equal to `height` x
   * `width` in row-major order.
   * - `pose`: The 2d pose of the person.
   */
  async segmentPersonParts(
      input: BodyPixInput,
      config: PersonInferenceConfig = PERSON_INFERENCE_CONFIG):
      Promise<PartSegmentation> {
    const configWithDefault:
        PersonInferenceConfig = {...PERSON_INFERENCE_CONFIG, ...config};
    validatePersonInferenceConfig(configWithDefault);
    const {partSegmentation, heatmapScores, offsets, padding} =
        this.segmentPersonPartsActivation(
            input, configWithDefault.segmentationThreshold);

    const [height, width] = partSegmentation.shape;
    const data = await partSegmentation.data() as Int32Array;
    partSegmentation.dispose();

    const pose = await decodeSinglePose(
        heatmapScores, offsets, this.baseModel.outputStride);

    const resultPose = scaleAndFlipPoses(
        [pose], [height, width], this.inputResolution, padding,
        configWithDefault.flipHorizontal)[0];

    heatmapScores.dispose();
    offsets.dispose();

    return {height, width, data, pose: resultPose};
  }

  /**
   * Given an image with multiple people, returns an *array* of PartSegmentation
   * object. Each element in the array corresponding to one
   * of the people in the input image. In other words, it predicts
   * instance-level multiple person body part segmentation & pose for each
   * person.
   *
   * This does standard ImageNet pre-processing before inferring through
   * the model. The image pixels should have values [0-255].
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
   * image to feed through the network.
   *
   * @param config MultiPersonInferenceConfig object that contains
   * parameters for the BodyPix inference using multi-person decoding.
   *
   * @return An array of PartSegmentation object, each containing a width,
   * height, a flattened array (with part id from 0-24 for the pixels that are
   * part of a corresponding body part, and -1 otherwise) and 2D pose. The width
   * and height correspond to the dimensions of the image. Each flattened part
   * segmentation array size is equal to `height` x `width`.
   */
  async segmentMultiPersonParts(
      input: BodyPixInput,
      config: MultiPersonInstanceInferenceConfig =
          MULTI_PERSON_INSTANCE_INFERENCE_CONFIG): Promise<PartSegmentation[]> {
    const configWithDefault: MultiPersonInstanceInferenceConfig = {
      ...MULTI_PERSON_INSTANCE_INFERENCE_CONFIG,
      ...config
    };
    validateMultiPersonInstanceInferenceConfig(configWithDefault);
    const [height, width] = getInputTensorDimensions(input);
    const inputResolution = this.inputResolution;
    const {resized, padding} = padAndResizeTo(input, inputResolution);
    const {
      segmentation,
      longOffsets,
      heatmapScoresRaw,
      offsetsRaw,
      displacementFwdRaw,
      displacementBwdRaw,
      partSegmentation,
    } = tf.tidy(() => {
      const {
        segmentLogits,
        longOffsets,
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd,
        partHeatmaps
      } = this.predictForMultiPersonInstanceSegmentationAndPart(resized);

      // decoding with scaling.
      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], inputResolution,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      // decoding with scaling.
      const scaledPartSegmentationScores = scaleAndCropToInputTensorShape(
          partHeatmaps, [height, width], inputResolution,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      const scaledLongOffsets = longOffsets;
      const segmentation = toMaskTensor(
          scaledSegmentScores.squeeze(),
          configWithDefault.segmentationThreshold);
      const partSegmentation =
          decodeOnlyPartSegmentation(scaledPartSegmentationScores);
      return {
        segmentation,
        longOffsets: scaledLongOffsets,
        heatmapScoresRaw: heatmapScores,
        offsetsRaw: offsets,
        displacementFwdRaw: displacementFwd,
        displacementBwdRaw: displacementBwd,
        partSegmentation
      };
    });

    const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
        await toTensorBuffers3D([
          heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw
        ]);

    let poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, this.baseModel.outputStride,
        config.maxDetections, config.scoreThreshold, config.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], inputResolution, padding, false);

    const instanceMasks = await decodeMultiplePartMasksGPU(
        segmentation, longOffsets, partSegmentation, poses, height, width,
        this.baseModel.outputStride, inputResolution,
        [[padding.top, padding.bottom], [padding.left, padding.right]],
        configWithDefault.scoreThreshold, configWithDefault.refineSteps,
        configWithDefault.minKeypointScore, configWithDefault.maxDetections);

    resized.dispose();
    segmentation.dispose();
    longOffsets.dispose();
    heatmapScoresRaw.dispose();
    offsetsRaw.dispose();
    displacementFwdRaw.dispose();
    displacementBwdRaw.dispose();
    partSegmentation.dispose();

    return instanceMasks;
  }

  public dispose() {
    this.baseModel.dispose();
  }
}

/**
 * Loads the MobileNet BodyPix model.
 */
async function loadMobileNet(config: ModelConfig): Promise<BodyPix> {
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  const multiplier = config.multiplier;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  const url = mobileNetSavedModel(outputStride, multiplier, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const mobilenet = new MobileNet(graphModel, outputStride);
  const validInputResolution = getValidInputResolutionDimensions(
      config.inputResolution, mobilenet.outputStride);
  return new BodyPix(mobilenet, validInputResolution);
}

/**
 * Loads the ResNet BodyPix model.
 */
async function loadResNet(config: ModelConfig): Promise<BodyPix> {
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  const url = resNet50SavedModel(outputStride, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const resnet = new ResNet(graphModel, outputStride);
  const validInputResolution = getValidInputResolutionDimensions(
      config.inputResolution, resnet.outputStride);
  return new BodyPix(resnet, validInputResolution);
}

/**
 * Loads the BodyPix model instance from a checkpoint, with the ResNet
 * or MobileNet architecture. The model to be loaded is configurable using the
 * config dictionary ModelConfig. Please find more details in the
 * documentation of the ModelConfig.
 *
 * @param config ModelConfig dictionary that contains parameters for
 * the BodyPix loading process. Please find more details of each parameters
 * in the documentation of the ModelConfig interface. The predefined
 * `MOBILENET_V1_CONFIG` and `RESNET_CONFIG` can also be used as references
 * for defining your customized config.
 */
export async function load(config: ModelConfig = MOBILENET_V1_CONFIG):
    Promise<BodyPix> {
  config = validateModelConfig(config);
  if (config.architecture === 'ResNet50') {
    return loadResNet(config);
  } else if (config.architecture === 'MobileNetV1') {
    return loadMobileNet(config);
  } else {
    return null;
  }
}
