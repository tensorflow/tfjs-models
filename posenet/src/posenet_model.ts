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

import * as tf from '@tensorflow/tfjs';

import {CheckpointLoader} from './checkpoint_loader';
import {checkpoints, resnet50_checkpoints} from './checkpoints';
import {assertValidOutputStride, MobileNet, MobileNetMultiplier, OutputStride, assertValidResolution} from './mobilenet';
import {ModelWeights} from './model_weights';
import {decodeMultiplePoses} from './multi_pose/decode_multiple_poses';
import ResNet from './resnet';
import {decodeSinglePose} from './single_pose/decode_single_pose';
import {Pose, PosenetInput} from './types';
import {getInputTensorDimensions, flipPoseHorizontal, flipPosesHorizontal, scalePose, scalePoses, toTensorBuffers3D, padAndResizeTo} from './util';

export type PoseNetResolution = 161|193|257|289|321|353|385|417|449|481|513;

export interface BackboneInterface {
  predict(input: tf.Tensor3D, outputStride: OutputStride): {[key: string]: tf.Tensor3D};
  dispose(): void;
  SUPPORTED_RESOLUTION: Array<PoseNetResolution>;
}

export class PoseNet {
  backbone: BackboneInterface;

  constructor(net: BackboneInterface) {
    this.backbone = net;
  }

  /**
   * Infer through PoseNet, and estimates a single pose using the outputs. This
   * does standard ImageNet pre-processing before inferring through the model.
   * The image should pixels should have values [0-255].
   * This method returns a single pose.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param imageScaleFactor A number between 0.2 and 1. Defaults to 0.50. What
   * to scale the image by before feeding it through the network.  Set this
   * number lower to scale down the image and increase the speed when feeding
   * through the network at the cost of accuracy.
   *
   * @param flipHorizontal.  Defaults to false.  If the poses should be
   * flipped/mirrored  horizontally.  This should be set to true for videos
   * where the video is by default flipped horizontally (i.e. a webcam), and you
   * want the poses to be returned in the proper orientation.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return A single pose with a confidence score, which contains an array of
   * keypoints indexed by part id, each with a score and position.  The
   * positions of the keypoints are in the same scale as the original image
   */
  async estimateSinglePose(
      input: PosenetInput, inputResolution = 513, flipHorizontal = false,
      outputStride: OutputStride = 32): Promise<Pose> {
        assertValidOutputStride(outputStride);
        assertValidResolution(inputResolution, outputStride);

        const [height, width] = getInputTensorDimensions(input);
        let [resizedHeight, resizedWidth] = [0, 0];
        let [padTop, padBottom, padLeft, padRight] = [0, 0, 0, 0];
        let heatmapScores, offsets;


        resizedHeight = inputResolution;
        resizedWidth = inputResolution;

        const outputs =
          tf.tidy(() => {
            const {resized, paddedBy} = padAndResizeTo(
              input, [resizedHeight, resizedWidth]);
            padTop = paddedBy[0][0];
            padBottom = paddedBy[0][1];
            padLeft = paddedBy[1][0];
            padRight = paddedBy[1][1];
            return this.backbone.predict(resized, outputStride);
          });

        heatmapScores = outputs.heatmapScores;
        offsets = outputs.offsets;

        const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);
        const scaleY = (height + padTop + padBottom) / (resizedHeight);
        const scaleX = (width + padLeft + padRight) / (resizedWidth);
        let scaledPose = scalePose(pose, scaleY, scaleX, -padTop, -padLeft);

        if (flipHorizontal) {
          scaledPose = flipPoseHorizontal(scaledPose, width)
        }

        heatmapScores.dispose();
        offsets.dispose();

        return scaledPose;
  }

  /**
   * Infer through PoseNet, and estimates multiple poses using the outputs.
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It detects
   * multiple poses and finds their parts from part scores and displacement
   * vectors using a fast greedy decoding algorithm.  It returns up to
   * `maxDetections` object instance detections in decreasing root score order.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param imageScaleFactor  A number between 0.2 and 1. Defaults to 0.50. What
   * to scale the image by before feeding it through the network.  Set this
   * number lower to scale down the image and increase the speed when feeding
   * through the network at the cost of accuracy.
   *
   * @param flipHorizontal Defaults to false.  If the poses should be
   * flipped/mirrored  horizontally.  This should be set to true for videos
   * where the video is by default flipped horizontally (i.e. a webcam), and you
   * want the poses to be returned in the proper orientation.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputSize - 1)/outputStride + 1
   *
   * @param maxDetections Maximum number of returned instance detections per
   * image. Defaults to 5.
   *
   * @param scoreThreshold Only return instance detections that have root part
   * score greater or equal to this value. Defaults to 0.5
   *
   * @param nmsRadius Non-maximum suppression part distance in pixels. It needs
   * to be strictly positive. Two parts suppress each other if they are less
   * than `nmsRadius` pixels away. Defaults to 20.
   *
   * @return An array of poses and their scores, each containing keypoints and
   * the corresponding keypoint scores.  The positions of the keypoints are
   * in the same scale as the original image
   */
  async estimateMultiplePoses(
      input: PosenetInput, inputResolution = 513, flipHorizontal = false,
      outputStride: OutputStride = 32, maxDetections = 5, scoreThreshold = .5,
      nmsRadius = 20): Promise<Pose[]> {
    assertValidOutputStride(outputStride);
    assertValidResolution(inputResolution, outputStride);

    const [height, width] = getInputTensorDimensions(input);
    let [resizedHeight, resizedWidth] = [0, 0];
    let [padTop, padBottom, padLeft, padRight] = [0, 0, 0, 0];
    let heatmapScores, offsets, displacementFwd, displacementBwd;

    resizedHeight = inputResolution;
    resizedWidth = inputResolution;

    const outputs =
      tf.tidy(() => {
        const {resized, paddedBy} = padAndResizeTo(
          input, [resizedHeight, resizedWidth]);
        padTop = paddedBy[0][0];
        padBottom = paddedBy[0][1];
        padLeft = paddedBy[1][0];
        padRight = paddedBy[1][1];
        return this.backbone.predict(resized, outputStride);
      });
    heatmapScores = outputs.heatmapScores;
    offsets = outputs.offsets;
    displacementFwd = outputs.displacementFwd;
    displacementBwd = outputs.displacementBwd;


    const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
        await toTensorBuffers3D(
            [heatmapScores, offsets, displacementFwd, displacementBwd]);

    const poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, outputStride,
        maxDetections, scoreThreshold,
        nmsRadius);

    const scaleY = (height + padTop + padBottom) / (resizedHeight);
    const scaleX = (width + padLeft + padRight) / (resizedWidth);
    let scaledPoses = scalePoses(poses, scaleY, scaleX, -padTop, -padLeft);

    if (flipHorizontal) {
      scaledPoses = flipPosesHorizontal(scaledPoses, width)
    }

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return scaledPoses;
  }

  public dispose() {
    this.backbone.dispose();
  }
}

/**
 * Loads the PoseNet model instance from a checkpoint, with the MobileNet
 * architecture specified by the multiplier.
 *
 * @param multiplier An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. Defaults to 1.01. It is the float multiplier for the depth (number
 of
 * channels) for all convolution ops. The value corresponds to a MobileNet
 * architecture and checkpoint.  The larger the value, the larger the size of
 * the layers, and more accurate the model at the cost of speed.  Set this to
 * a smaller value to increase speed at the cost of accuracy.
 *
 */
export async function loadMobileNet(
  multiplier: MobileNetMultiplier = 1.01):
    Promise<PoseNet> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }
  // TODO: figure out better way to decide below.
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'number',
      () => `got multiplier type of ${typeof multiplier} when it should be a ` +
          `number.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier.toString()) >= 0,
      () => `invalid multiplier value of ${
                multiplier}.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  const mobileNet: MobileNet = await mobilenetLoader.load(multiplier);

  return new PoseNet(mobileNet);
}

export const mobilenetLoader = {
  load: async(multiplier: MobileNetMultiplier): Promise<MobileNet> => {
    const checkpoint = checkpoints[multiplier];

    const checkpointLoader = new CheckpointLoader(checkpoint.url);

    const variables = await checkpointLoader.getAllVariables();

    const weights = new ModelWeights(variables);

    return new MobileNet(weights, checkpoint.architecture);
  },

};

/**
 * Loads the PoseNet model instance from a checkpoint, with the ResNet
 * architecture.
 *
 * @param outputStride Specifies the output stride of the ResNet model.
 * The smaller the value, the larger the output resolution, and more accurate the model
 * at the cost of speed.  Set this to a larger value to increase speed at the cost of accuracy.
 * Currently only 32 is supported for ResNet.
 *
 * @param resolution Specifies the input resolution of the ResNet model.
 * The larger the value, more accurate the model at the cost of speed.
 * Set this to a smaller value to increase speed at the cost of accuracy.
 * Currently only input resolution 257 and 513 are supported for ResNet.
 *
 */
export async function loadResNet(outputStride: OutputStride, resolution: PoseNetResolution):
    Promise<PoseNet> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }

  tf.util.assert(
      [32].indexOf(outputStride) >= 0,
    () => `invalid stride value of ${outputStride}.  No checkpoint exists for that ` +
        `stride. Currently must be one of [32].`);

  tf.util.assert(
    [513, 257].indexOf(resolution) >= 0,
    () => `invalid resolution value of ${resolution}.  No checkpoint exists for that ` +
        `resolution. Currently must be one of [513, 257].`);

  const graphModel = await tf.loadGraphModel(resnet50_checkpoints[resolution][outputStride]);
  const resnet = new ResNet(graphModel, outputStride)
  return new PoseNet(resnet);
}

export async function load(architecture: string,
   outputStride: OutputStride = 32,
   resolution: PoseNetResolution = 513): Promise<PoseNet> {
  if (architecture.includes('ResNet50')) {
    return loadResNet(outputStride, resolution);
  } else {
    const multiplier = architecture.split(' ')[1];
    return loadMobileNet(+multiplier as MobileNetMultiplier);
  }
}

