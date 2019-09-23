
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
import {decodeOnlyPartSegmentation, decodePartSegmentation, toMask} from './decode_part_map';
import {MobileNetMultiplier} from './mobilenet';
import {MobileNet} from './mobilenet';
import {decodeMultipleMasksGPU, decodeMultiplePartMasksGPU} from './multi_person/decode_multiple_masks';
import {decodeMultiplePoses} from './multi_person/decode_multiple_poses';
import {ResNet} from './resnet';
import {decodeSinglePose} from './sinlge_person/decode_single_pose';
import {BodyPixInput, Padding, PartSegmentation, PersonSegmentation} from './types';
import {getInputTensorDimensions, padAndResizeTo, scaleAndCropToInputTensorShape, scaleAndFlipPoses, toInputTensor, toTensorBuffers3D} from './util';

export type BodyPixInputResolution =
    161|193|257|289|321|353|385|417|449|481|513|801|1217;
export type BodyPixOutputStride = 32|16|8;
export type BodyPixArchitecture = 'ResNet50'|'MobileNetV1';
export type BodyPixDecodingMethod = 'single-person'|'multi-person';
export type BodyPixQuantBytes = 1|2|4;

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
   * heatmapScores: A Tensor3D that represents the heatmapScores.
   * offsets: A Tensor3D that represents the offsets.
   * displacementFwd: A Tensor3D that represents the forward displacement.
   * displacementBwd: A Tensor3D that represents the backward displacement.
   * segmentation: A Tensor3D that represents the segmentation of all people.
   * longOffsets: A Tensor3D that represents the long offsets used for instance
   * grouping. partHeatmaps: A Tensor3D that represents the body part
   * segmentation. partOffsets: A Tensor3D that represents the UV offsets within
   * each body part.
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
 * `architecture`: PoseNetArchitecture. It determines wich PoseNet architecture
 * to load. The supported architectures are: MobileNetV1 and ResNet.
 *
 * `outputStride`: Specifies the output stride of the PoseNet model.
 * The smaller the value, the larger the output resolution, and more accurate
 * the model at the cost of speed.  Set this to a larger value to increase speed
 * at the cost of accuracy. Stride 32 is supported for ResNet and
 * stride 8,16,32 are supported for various MobileNetV1 models.
 *
 * `inputResolution`: One of the number specified by BodyPixInputResolution.
 * It represents the input image resolution of the model. The larger the size
 * of the input image, and more accurate the model at the cost of speed.
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
  architecture: BodyPixArchitecture;
  outputStride: BodyPixOutputStride;
  inputResolution: BodyPixInputResolution;
  multiplier?: MobileNetMultiplier;
  modelUrl?: string;
  quantBytes?: BodyPixQuantBytes;
}

// The default configuration for loading MobileNetV1 based BodyPix.
//
// TODO(tylerzhu): Adds MobileNetV1 BodyPix 2.0 configuration.
//
// (And for references, the default configuration for loading ResNet
// based PoseNet is also included).
const RESNET_CONFIG = {
  architecture: 'ResNet50',
  outputStride: 32,
  inputResolution: 513,
  quantBytes: 4,
} as ModelConfig;

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
 * the intermediate person segmentation probability. The probablity of each
 * pixel belongs to a person is in range [0, 1]. If the probablity is greater
 * than the `segmentationThreshold`, it will be set to 1 otherwise 0.
 *
 */
export interface InferenceConfig {
  flipHorizontal: boolean;
  segmentationThreshold: number;
}

/**
 * Single Person Inference Config
 */
export interface SinglePersonInferenceConfig extends InferenceConfig {}

/**
 * Multiple Person Inference Config
 *
 * `maxDetections`: Maximum number of returned instance detections per image.
 * Defaults to 10
 *
 * `scoreThreshold`: Only return instance detections that have root part
 * score greater or equal to this value. Defaults to 0.5
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
 **/
export interface MultiPersonInferenceConfig extends InferenceConfig {
  maxDetections?: number;
  scoreThreshold?: number;
  nmsRadius?: number;
  minKeypointScore?: number;
  refineSteps?: number;
}

export const SINGLE_PERSON_INFERENCE_CONFIG: SinglePersonInferenceConfig = {
  flipHorizontal: false,
  segmentationThreshold: 0.7
};

export const MULTI_PERSON_INFERENCE_CONFIG: MultiPersonInferenceConfig = {
  flipHorizontal: false,
  segmentationThreshold: 0.7,
  maxDetections: 10,
  scoreThreshold: 0.2,
  nmsRadius: 20,
  minKeypointScore: 0.3,
  refineSteps: 10
};

function validateSinglePersonInferenceConfig(
    config: SinglePersonInferenceConfig) {
  const segmentationThreshold = config.segmentationThreshold;
  if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
    throw new Error(
        `segmentationThreshold ${segmentationThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }
}

function validateMultiPersonInferenceConfig(
    config: MultiPersonInferenceConfig) {
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
  baseModel: BaseModel;
  inputResolution: BodyPixInputResolution;

  constructor(net: BaseModel, inputResolution: BodyPixInputResolution) {
    this.baseModel = net;
    this.inputResolution = inputResolution;
  }

  predictForSinglePersonSegmentationLogits(input: tf.Tensor3D): {
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
      segmentLogits: segmentation, heatmapScores: heatmapScores,
          offsets: offsets, displacementFwd: displacementFwd,
          displacementBwd: displacementBwd,
    }
  }

  predictForSinglePersonPartMapLogits(input: tf.Tensor3D): {
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
      segmentLogits: segmentation, partHeatmapLogits: partHeatmaps,
          heatmapScores: heatmapScores, offsets: offsets,
          displacementFwd: displacementFwd, displacementBwd: displacementBwd,
    }
  }

  predictForMultiPersonSegmentationAndPartMap(input: tf.Tensor3D): {
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
      segmentLogits: segmentation, longOffsets: longOffsets,
          heatmapScores: heatmapScores, offsets: offsets,
          displacementFwd: displacementFwd, displacementBwd: displacementBwd,
          partHeatmaps: partHeatmaps
    }
  }

  /**
   * Given an image with a person, returns a dictioanry of all intermediate
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
  estimateSinglePersonSegmentationActivation(
      input: BodyPixInput, segmentationThreshold = 0.5): {
    segmentation: tf.Tensor2D,
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    padding: Padding
  } {
    const inputResolution = this.inputResolution;
    const imageTensor = toInputTensor(input);
    const {resized, padding} =
        padAndResizeTo(imageTensor, [inputResolution, inputResolution]);

    const {segmentation, heatmapScores, offsets} = tf.tidy(() => {
      const {
        segmentLogits,
        heatmapScores,
        offsets,
      } = this.predictForSinglePersonSegmentationLogits(resized);

      const [resizedHeight, resizedWidth] = resized.shape;
      const [height, width] = imageTensor.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true);

      return {
        segmentation:
            toMask(scaledSegmentScores.squeeze(), segmentationThreshold),
        heatmapScores,
        offsets,
      };
    });
    resized.dispose();
    return {
      segmentation, heatmapScores, offsets, padding
    }
  }

  /**
   * Given an image with a person, returns a PersonSegmentation dictionary that
   * contains the segmentation mask and the pose for the person.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config SinglePersonInferenceConfig object that contains
   * parameters for the BodyPix inference using single person decoding.
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
  async estimateSinglePersonSegmentation(
      input: BodyPixInput,
      config: SinglePersonInferenceConfig = SINGLE_PERSON_INFERENCE_CONFIG):
      Promise<PersonSegmentation> {
    const configWithDefault: SinglePersonInferenceConfig = {
      ...SINGLE_PERSON_INFERENCE_CONFIG,
      ...config
    };
    validateSinglePersonInferenceConfig(configWithDefault);
    const {segmentation, heatmapScores, offsets, padding} =
        this.estimateSinglePersonSegmentationActivation(
            input, configWithDefault.segmentationThreshold);

    const [height, width] = segmentation.shape;

    const result = await segmentation.data() as Uint8Array;
    segmentation.dispose();

    const pose = await decodeSinglePose(
        heatmapScores, offsets, this.baseModel.outputStride);

    const resultPose = scaleAndFlipPoses(
        [pose], [height, width], [this.inputResolution, this.inputResolution],
        padding, configWithDefault.flipHorizontal)[0];

    heatmapScores.dispose();
    offsets.dispose();

    return {height, width, data: result, pose: resultPose};
  }

  /**
   * Given an image with multiple people, returns an array of PersonSegmentation
   * object. This does standard ImageNet pre-processing before inferring through
   * the model.  The image pixels should have values [0-255].
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
  async estimateMultiplePersonSegmentation(
      input: BodyPixInput,
      config: MultiPersonInferenceConfig = MULTI_PERSON_INFERENCE_CONFIG):
      Promise<PersonSegmentation[]> {
    const configWithDefault: MultiPersonInferenceConfig = {
      ...MULTI_PERSON_INFERENCE_CONFIG,
      ...config
    };
    validateMultiPersonInferenceConfig(configWithDefault);
    const [height, width] = getInputTensorDimensions(input);
    const inputResolution = this.inputResolution;

    const {resized, padding} =
        padAndResizeTo(input, [inputResolution, inputResolution]);
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
      } = this.predictForMultiPersonSegmentationAndPartMap(resized);
      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [inputResolution, inputResolution],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true);
      const longOffsetsResized = false;
      let scaledLongOffsets;
      if (longOffsetsResized) {
        scaledLongOffsets = scaleAndCropToInputTensorShape(
            longOffsets, [height, width], [inputResolution, inputResolution],
            [[padding.top, padding.bottom], [padding.left, padding.right]],
            true);
      } else {
        scaledLongOffsets = longOffsets;
      }

      const segmentation = toMask(
          scaledSegmentScores.squeeze(),
          configWithDefault.segmentationThreshold);

      return {
        segmentation: segmentation,
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
        displacementsBwdBuffer, this.baseModel.outputStride, 5,
        configWithDefault.scoreThreshold, configWithDefault.nmsRadius);

    poses = scaleAndFlipPoses(
        poses, [height, width], [inputResolution, inputResolution], padding,
        false);

    const instanceMasks = decodeMultipleMasksGPU(
        segmentation, longOffsets, poses, height, width,
        this.baseModel.outputStride, [inputResolution, inputResolution],
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
   * Given an image with a person, returns a dictionary containing: height,
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
  estimateSinglePersonPartSegmentationActivation(
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
    } = padAndResizeTo(imageTensor, [inputResolution, inputResolution]);

    const {partSegmentation, heatmapScores, offsets} = tf.tidy(() => {
      const {segmentLogits, partHeatmapLogits, heatmapScores, offsets} =
          this.predictForSinglePersonPartMapLogits(resized);

      const [resizedHeight, resizedWidth] = resized.shape;
      const [height, width] = imageTensor.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true);

      const scaledPartHeatmapScore = scaleAndCropToInputTensorShape(
          partHeatmapLogits, [height, width], [resizedHeight, resizedWidth],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true);
      const segmentation =
          toMask(scaledSegmentScores.squeeze(), segmentationThreshold);
      return {
        partSegmentation:
            decodePartSegmentation(segmentation, scaledPartHeatmapScore),
        heatmapScores,
        offsets
      };
    });
    resized.dispose();
    return {
      partSegmentation, heatmapScores, offsets, padding
    }
  }

  /**
   * Given an image with a person, returns a PartSegmentation dictionary that
   * contains the person part segmentation mask and the pose for the person.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param config SinglePersonInferenceConfig object that contains
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
  async estimateSinglePersonPartSegmentation(
      input: BodyPixInput,
      config: SinglePersonInferenceConfig = SINGLE_PERSON_INFERENCE_CONFIG):
      Promise<PartSegmentation> {
    const configWithDefault: SinglePersonInferenceConfig = {
      ...SINGLE_PERSON_INFERENCE_CONFIG,
      ...config
    };
    validateSinglePersonInferenceConfig(configWithDefault);
    const {partSegmentation, heatmapScores, offsets, padding} =
        this.estimateSinglePersonPartSegmentationActivation(
            input, configWithDefault.segmentationThreshold);

    const [height, width] = partSegmentation.shape;
    const data = await partSegmentation.data() as Int32Array;
    partSegmentation.dispose();

    const pose = await decodeSinglePose(
        heatmapScores, offsets, this.baseModel.outputStride);

    const resultPose = scaleAndFlipPoses(
        [pose], [height, width], [this.inputResolution, this.inputResolution],
        padding, configWithDefault.flipHorizontal)[0];

    heatmapScores.dispose();
    offsets.dispose();

    return {height, width, data, pose: resultPose};
  }

  /**
   * Given an image with multiple people, returns an array of PartSegmentation
   * object. This does standard ImageNet pre-processing before inferring through
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
  async estimateMultiplePersonPartSegmentation(
      input: BodyPixInput,
      config: MultiPersonInferenceConfig = MULTI_PERSON_INFERENCE_CONFIG):
      Promise<PartSegmentation[]> {
    const configWithDefault: MultiPersonInferenceConfig = {
      ...MULTI_PERSON_INFERENCE_CONFIG,
      ...config
    };
    validateMultiPersonInferenceConfig(configWithDefault);
    const [height, width] = getInputTensorDimensions(input);
    const inputResolution = this.inputResolution;
    const {resized, padding} =
        padAndResizeTo(input, [inputResolution, inputResolution]);
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
      } = this.predictForMultiPersonSegmentationAndPartMap(resized);

      // decoding with scaling.
      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentLogits, [height, width], [inputResolution, inputResolution],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true);

      // decoding with scaling.
      const scaledPartSegmentationScores = scaleAndCropToInputTensorShape(
          partHeatmaps, [height, width], [inputResolution, inputResolution],
          [[padding.top, padding.bottom], [padding.left, padding.right]], true)

      const scaledLongOffsets = longOffsets;
      const segmentation = toMask(
          scaledSegmentScores.squeeze(),
          configWithDefault.segmentationThreshold);
      const partSegmentation =
          decodeOnlyPartSegmentation(scaledPartSegmentationScores);
      return {
        segmentation: segmentation,
        longOffsets: scaledLongOffsets,
        heatmapScoresRaw: heatmapScores,
        offsetsRaw: offsets,
        displacementFwdRaw: displacementFwd,
        displacementBwdRaw: displacementBwd,
        partSegmentation: partSegmentation
      };
    });

    const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
        await toTensorBuffers3D([
          heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw
        ]);

    let poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, this.baseModel.outputStride, 30, 0.3, 20);

    poses = scaleAndFlipPoses(
        poses, [height, width], [inputResolution, inputResolution], padding,
        false);

    const instanceMasks = decodeMultiplePartMasksGPU(
        segmentation, longOffsets, partSegmentation, poses, height, width,
        this.baseModel.outputStride, [inputResolution, inputResolution],
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

  const url = mobileNetCheckpoint(outputStride, multiplier, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const mobilenet = new MobileNet(graphModel, outputStride);
  return new BodyPix(mobilenet, config.inputResolution);
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

  const url = resNet50Checkpoint(outputStride, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const resnet = new ResNet(graphModel, outputStride);
  return new BodyPix(resnet, config.inputResolution);
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
export async function load(config: ModelConfig = RESNET_CONFIG):
    Promise<BodyPix> {
  // config = validateModelConfig(config);
  if (config.architecture === 'ResNet50') {
    return loadResNet(config);
  } else if (config.architecture === 'MobileNetV1') {
    return loadMobileNet(config);
  } else {
    return null;
  }
}
