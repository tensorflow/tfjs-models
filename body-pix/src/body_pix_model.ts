
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

import {BaseModel} from './base_model';
import {decodeOnlyPartSegmentation, decodePartSegmentation, toMaskTensor} from './decode_part_map';
import {MobileNet} from './mobilenet';
import {decodePersonInstanceMasks, decodePersonInstancePartMasks} from './multi_person/decode_instance_masks';
import {decodeMultiplePoses} from './multi_person/decode_multiple_poses';
import {ResNet} from './resnet';
import {mobileNetSavedModel, resNet50SavedModel} from './saved_models';
import {BodyPixArchitecture, BodyPixInput, BodyPixInternalResolution, BodyPixMultiplier, BodyPixOutputStride, BodyPixQuantBytes, Padding} from './types';
import {PartSegmentation, PersonSegmentation, SemanticPartSegmentation, SemanticPersonSegmentation} from './types';
import {getInputSize, padAndResizeTo, scaleAndCropToInputTensorShape, scaleAndFlipPoses, toInputResolutionHeightAndWidth, toTensorBuffers3D} from './util';

const APPLY_SIGMOID_ACTIVATION = true;
const FLIP_POSES_AFTER_SCALING = false;

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
  multiplier?: BodyPixMultiplier;
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
//   quantBytes: 4,
// } as ModelConfig;
// ```

const MOBILENET_V1_CONFIG = {
  architecture: 'MobileNetV1',
  outputStride: 16,
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

function validateModelConfig(config: ModelConfig): ModelConfig {
  config = config || MOBILENET_V1_CONFIG;

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
 * `internalResolution`: Defaults to 'medium'. The internal resolution
 * percentage that the input is resized to before inference. The larger the
 * internalResolution the more accurate the model at the cost of slower
 * prediction times. Available values are 'low', 'medium', 'high', 'full', or a
 * percentage value between 0 and 1. The values 'low', 'medium', 'high', and
 * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
 *
 * `segmentationThreshold`: The minimum that segmentation values must
 * have to be considered part of the person. Affects the generation of the
 * segmentation mask. More specifically, it is the threshold used to binarize
 * the intermediate person segmentation probability. The probability of each
 * pixel belongs to a person is in range [0, 1]. If the probability is greater
 * than the `segmentationThreshold`, it will be set to 1 otherwise 0.
 *
 */
export interface InferenceConfig {
  flipHorizontal?: boolean;
  internalResolution?: BodyPixInternalResolution;
  segmentationThreshold?: number;
}

/**
 * Person Inference Config
 *
 * `maxDetections`: Defaults to 10. Maximum number of person pose detections per
 * image.
 *
 * `scoreThreshold`: Defaults to 0.4. Only return person pose that have root
 * part score greater or equal to this value.
 *
 * `nmsRadius`: Defaults to 20. Non-maximum suppression part distance in pixels.
 * It needs to be strictly positive. Two pose keypoints suppress each other if
 * they are less than `nmsRadius` pixels away.
 */
export interface PersonInferenceConfig extends InferenceConfig {
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
}

/**
 * Multiple Person Instance Inference Config
 *
 * `maxDetections`: Defaults to 10. Maximum number of returned instance
 * segmentation and pose detections per image.
 *
 * `scoreThreshold`: Defaults to 0.4. Only returns and uses person
 * poses for instance segmentation assignment when the pose has root part score
 * greater or equal to this value.
 *
 * `nmsRadius`: Defaults to 20. Non-maximum suppression part distance in pixels.
 * It needs to be strictly positive. Two parts suppress each other if they are
 * less than `nmsRadius` pixels away.
 *
 * `minKeypointScore`: Default to 0.3. Keypoints above the score are used
 * for matching and assigning segmentation mask to each person.
 *
 * `refineSteps`: Default to 10. The number of refinement steps used when
 * assigning the instance segmentation. It needs to be strictly positive. The
 * larger the higher the accuracy and slower the inference.
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
  internalResolution: 'medium',
  segmentationThreshold: 0.7,
  maxDetections: 10,
  scoreThreshold: 0.4,
  nmsRadius: 20,
};

export const MULTI_PERSON_INSTANCE_INFERENCE_CONFIG:
    MultiPersonInstanceInferenceConfig = {
      flipHorizontal: false,
      internalResolution: 'medium',
      segmentationThreshold: 0.7,
      maxDetections: 10,
      scoreThreshold: 0.4,
      nmsRadius: 20,
      minKeypointScore: 0.3,
      refineSteps: 10
    };

function validatePersonInferenceConfig(config: PersonInferenceConfig) {
  const {segmentationThreshold, maxDetections, scoreThreshold, nmsRadius} =
      config;

  if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
    throw new Error(
        `segmentationThreshold ${segmentationThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }

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

function validateMultiPersonInstanceInferenceConfig(
    config: MultiPersonInstanceInferenceConfig) {
  const {
    segmentationThreshold,
    maxDetections,
    scoreThreshold,
    nmsRadius,
    minKeypointScore,
    refineSteps
  } = config;

  if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
    throw new Error(
        `segmentationThreshold ${segmentationThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }

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

  if (refineSteps <= 0 || refineSteps > 20) {
    throw new Error(
        `Invalid refineSteps ${refineSteps}.` +
        `Should be in range [1, 20]`);
  }
}

export class BodyPix {
  baseModel: BaseModel;

  constructor(net: BaseModel) {
    this.baseModel = net;
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
   * @param internalResolution Defaults to 'medium'. The internal resolution
   * that the input is resized to before inference. The larger the
   * internalResolution the more accurate the model at the cost of slower
   * prediction times. Available values are 'low', 'medium', 'high', 'full', or
   * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
   * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
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
   * pose estimation decoding.
   * - `offsets`: A 3d Tensor of the keypoint offsets used by pose
   * estimation decoding.
   * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement used
   * by pose estimation decoding.
   * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
   * by pose estimation decoding.
   * - `padding`: The padding (unit pixels) being applied to the input image
   * before it is fed into the model.
   */
  segmentPersonActivation(
      input: BodyPixInput, internalResolution: BodyPixInternalResolution,
      segmentationThreshold = 0.5): {
    segmentation: tf.Tensor2D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
    padding: Padding,
    internalResolutionHeightAndWidth: [number, number]
  } {
    const [height, width] = getInputSize(input);
    const internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(
        internalResolution, this.baseModel.outputStride, [height, width]);
    const {resized, padding} =
        padAndResizeTo(input, internalResolutionHeightAndWidth);

    const {
      segmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd
    } = tf.tidy(() => {
      const {
        segmentLogits,
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd
      } = this.predictForPersonSegmentation(resized);

      const [resizedHeight, resizedWidth] = resized.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      return {
        segmentation:
            toMaskTensor(scaledSegmentScores.squeeze(), segmentationThreshold),
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd,
      };
    });
    resized.dispose();
    return {
      segmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      padding,
      internalResolutionHeightAndWidth
    };
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
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config PersonInferenceConfig object that contains
   * parameters for the BodyPix inference using person decoding.
   *
   * @return A SemanticPersonSegmentation dictionary that contains height,
   * width, the flattened binary segmentation mask and the poses for all people.
   * The width and height correspond to the same dimensions of the input image.
   * - `height`: The height of the segmentation data in pixel unit.
   * - `width`: The width of the segmentation data in pixel unit.
   * - `data`: The flattened Uint8Array of segmentation data. 1 means the pixel
   * belongs to a person and 0 means the pixel doesn't belong to a person. The
   * size of the array is equal to `height` x `width` in row-major order.
   * - `allPoses`: The 2d poses of all people.
   */
  async segmentPerson(
      input: BodyPixInput,
      config: PersonInferenceConfig = PERSON_INFERENCE_CONFIG):
      Promise<SemanticPersonSegmentation> {
    config = {...PERSON_INFERENCE_CONFIG, ...config};

    validatePersonInferenceConfig(config);

    const {
      segmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      padding,
      internalResolutionHeightAndWidth
    } =
        this.segmentPersonActivation(
            input, config.internalResolution, config.segmentationThreshold);

    const [height, width] = segmentation.shape;

    const result = await segmentation.data() as Uint8Array;
    segmentation.dispose();

    const tensorBuffers = await toTensorBuffers3D(
        [heatmapScores, offsets, displacementFwd, displacementBwd]);
    const [scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf] =
        tensorBuffers;

    let poses = decodeMultiplePoses(
        scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf,
        this.baseModel.outputStride, config.maxDetections,
        config.scoreThreshold, config.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], internalResolutionHeightAndWidth, padding,
        FLIP_POSES_AFTER_SCALING);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return {height, width, data: result, allPoses: poses};
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
    config = {...MULTI_PERSON_INSTANCE_INFERENCE_CONFIG, ...config};
    validateMultiPersonInstanceInferenceConfig(config);
    const [height, width] = getInputSize(input);
    const internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(
        config.internalResolution, this.baseModel.outputStride,
        [height, width]);

    const {resized, padding} =
        padAndResizeTo(input, internalResolutionHeightAndWidth);
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
          segmentLogits, [height, width], internalResolutionHeightAndWidth,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);
      const longOffsetsResized = false;
      let scaledLongOffsets;
      if (longOffsetsResized) {
        scaledLongOffsets = scaleAndCropToInputTensorShape(
            longOffsets, [height, width], internalResolutionHeightAndWidth,
            [[padding.top, padding.bottom], [padding.left, padding.right]],
            APPLY_SIGMOID_ACTIVATION);
      } else {
        scaledLongOffsets = longOffsets;
      }

      const segmentation = toMaskTensor(
          scaledSegmentScores.squeeze(), config.segmentationThreshold);

      return {
        segmentation,
        longOffsets: scaledLongOffsets,
        heatmapScoresRaw: heatmapScores,
        offsetsRaw: offsets,
        displacementFwdRaw: displacementFwd,
        displacementBwdRaw: displacementBwd,
      };
    });

    const tensorBuffers = await toTensorBuffers3D(
        [heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw]);
    const [scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf] =
        tensorBuffers;

    let poses = decodeMultiplePoses(
        scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf,
        this.baseModel.outputStride, config.maxDetections,
        config.scoreThreshold, config.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], internalResolutionHeightAndWidth, padding,
        FLIP_POSES_AFTER_SCALING);

    const instanceMasks = await decodePersonInstanceMasks(
        segmentation, longOffsets, poses, height, width,
        this.baseModel.outputStride, internalResolutionHeightAndWidth, padding,
        config.scoreThreshold, config.refineSteps, config.minKeypointScore,
        config.maxDetections);

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
   * @param internalResolution Defaults to 'medium'. The internal resolution
   * percentage that the input is resized to before inference. The larger the
   * internalResolution the more accurate the model at the cost of slower
   * prediction times. Available values are 'low', 'medium', 'high', 'full', or
   * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
   * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
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
   * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement
   * used by pose estimation decoding.
   * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
   * by pose estimation decoding.
   * - `padding`: The padding (unit pixels) being applied to the input image
   * before it is fed into the model.
   */
  segmentPersonPartsActivation(
      input: BodyPixInput, internalResolution: BodyPixInternalResolution,
      segmentationThreshold = 0.5): {
    partSegmentation: tf.Tensor2D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D,
    padding: Padding,
    internalResolutionHeightAndWidth: [number, number]
  } {
    const [height, width] = getInputSize(input);
    const internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(
        internalResolution, this.baseModel.outputStride, [height, width]);
    const {
      resized,
      padding,
    } = padAndResizeTo(input, internalResolutionHeightAndWidth);

    const {
      partSegmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd
    } = tf.tidy(() => {
      const {
        segmentLogits,
        partHeatmapLogits,
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd
      } = this.predictForPersonSegmentationAndPart(resized);

      const [resizedHeight, resizedWidth] = resized.shape;

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
        offsets,
        displacementFwd,
        displacementBwd,
      };
    });
    resized.dispose();
    return {
      partSegmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      padding,
      internalResolutionHeightAndWidth
    };
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
   * @return A SemanticPartSegmentation dictionary that contains height, width,
   * the flattened binary segmentation mask and the pose for the person. The
   * width and height correspond to the same dimensions of the input image.
   * - `height`: The height of the person part segmentation data in pixel unit.
   * - `width`: The width of the person part segmentation data in pixel unit.
   * - `data`: The flattened Int32Array of person part segmentation data with a
   * part id from 0-24 for the pixels that are part of a corresponding body
   * part, and -1 otherwise. The size of the array is equal to `height` x
   * `width` in row-major order.
   * - `allPoses`: The 2d poses of all people.
   */
  async segmentPersonParts(
      input: BodyPixInput,
      config: PersonInferenceConfig = PERSON_INFERENCE_CONFIG):
      Promise<SemanticPartSegmentation> {
    config = {...PERSON_INFERENCE_CONFIG, ...config};

    validatePersonInferenceConfig(config);
    const {
      partSegmentation,
      heatmapScores,
      offsets,
      displacementFwd,
      displacementBwd,
      padding,
      internalResolutionHeightAndWidth
    } =
        this.segmentPersonPartsActivation(
            input, config.internalResolution, config.segmentationThreshold);

    const [height, width] = partSegmentation.shape;
    const data = await partSegmentation.data() as Int32Array;
    partSegmentation.dispose();

    const tensorBuffers = await toTensorBuffers3D(
        [heatmapScores, offsets, displacementFwd, displacementBwd]);
    const [scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf] =
        tensorBuffers;

    let poses = decodeMultiplePoses(
        scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf,
        this.baseModel.outputStride, config.maxDetections,
        config.scoreThreshold, config.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], internalResolutionHeightAndWidth, padding,
        FLIP_POSES_AFTER_SCALING);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return {height, width, data, allPoses: poses};
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
    config = {...MULTI_PERSON_INSTANCE_INFERENCE_CONFIG, ...config};

    validateMultiPersonInstanceInferenceConfig(config);
    const [height, width] = getInputSize(input);
    const internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(
        config.internalResolution, this.baseModel.outputStride,
        [height, width]);
    const {resized, padding} =
        padAndResizeTo(input, internalResolutionHeightAndWidth);
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
          segmentLogits, [height, width], internalResolutionHeightAndWidth,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      // decoding with scaling.
      const scaledPartSegmentationScores = scaleAndCropToInputTensorShape(
          partHeatmaps, [height, width], internalResolutionHeightAndWidth,
          [[padding.top, padding.bottom], [padding.left, padding.right]],
          APPLY_SIGMOID_ACTIVATION);

      const scaledLongOffsets = longOffsets;
      const segmentation = toMaskTensor(
          scaledSegmentScores.squeeze(), config.segmentationThreshold);
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

    const tensorBuffers = await toTensorBuffers3D(
        [heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw]);
    const [scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf] =
        tensorBuffers;

    let poses = decodeMultiplePoses(
        scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf,
        this.baseModel.outputStride, config.maxDetections,
        config.scoreThreshold, config.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], internalResolutionHeightAndWidth, padding,
        FLIP_POSES_AFTER_SCALING);

    const instanceMasks = await decodePersonInstancePartMasks(
        segmentation, longOffsets, partSegmentation, poses, height, width,
        this.baseModel.outputStride, internalResolutionHeightAndWidth, padding,
        config.scoreThreshold, config.refineSteps, config.minKeypointScore,
        config.maxDetections);

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
  return new BodyPix(mobilenet);
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
  return new BodyPix(resnet);
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
