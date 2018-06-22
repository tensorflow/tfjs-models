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
import {checkpoints} from './checkpoints';
// tslint:disable-next-line:max-line-length
import {assertValidOutputStride, assertValidScaleFactor, MobileNet, MobileNetMultiplier, OutputStride} from './mobilenet';
import {decodeMultiplePoses} from './multiPose/decodeMultiplePoses';
import {decodeSinglePose} from './singlePose/decodeSinglePose';
import {Pose} from './types';
import {getValidResolution, scalePose, scalePoses} from './util';

export type PoseNetResolution = 161|193|257|289|321|353|385|417|449|481|513;

export type InputType =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;

function toInputTensor(
    input: InputType, resizeHeight: number, resizeWidth: number,
    flipHorizontal: boolean): tf.Tensor3D {
  const imageTensor = input instanceof tf.Tensor ? input : tf.fromPixels(input);

  if (flipHorizontal) {
    return imageTensor.reverse(1).resizeBilinear([resizeHeight, resizeWidth]);
  } else {
    return imageTensor.resizeBilinear([resizeHeight, resizeWidth]);
  }
}

export class PoseNet {
  mobileNet: MobileNet;

  constructor(mobileNet: MobileNet) {
    this.mobileNet = mobileNet;
  }

  /**
   * Infer through PoseNet. This does standard ImageNet pre-processing before
   * inferring through the model. The image should pixels should have values
   * [0-255]. This method returns the heatmaps and offsets.  Infers through the
   * outputs that are needed for single pose decoding
   *
   * @param input un-preprocessed input image, with values in range [0-255]
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16.  The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return heatmapScores, offsets
   */
  predictForSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 16):
      {heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D} {
    assertValidOutputStride(outputStride);
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      const heatmaps =
          this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');

      const offsets = this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');

      return {heatmapScores: heatmaps.sigmoid(), offsets};
    });
  }

  /**
   * Infer through PoseNet. This does standard ImageNet pre-processing before
   * inferring through the model. The image should pixels should have values
   * [0-255]. Infers through the outputs that are needed for multiple pose
   * decoding. This method returns the heatmaps offsets, and mid-range
   * displacements.
   *
   * @param input un-preprocessed input image, with values in range [0-255]
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return heatmapScores, offsets, displacementFwd, displacementBwd
   */
  predictForMultiPose(input: tf.Tensor3D, outputStride: OutputStride = 16): {
    heatmapScores: tf.Tensor3D,
    offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D,
    displacementBwd: tf.Tensor3D
  } {
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      const heatmaps =
          this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');

      const offsets = this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');

      const displacementFwd =
          this.mobileNet.convToOutput(mobileNetOutput, 'displacement_fwd_2');

      const displacementBwd =
          this.mobileNet.convToOutput(mobileNetOutput, 'displacement_bwd_2');

      return {
        heatmapScores: heatmaps.sigmoid(),
        offsets,
        displacementFwd,
        displacementBwd
      };
    });
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
      input: InputType, imageScaleFactor = 0.5, flipHorizontal = false,
      outputStride: OutputStride = 16): Promise<Pose> {
    assertValidOutputStride(outputStride);
    assertValidScaleFactor(imageScaleFactor);

    const [height, width] = input instanceof tf.Tensor ?
        [input.shape[0], input.shape[1]] :
        [input.height, input.width];
    const resizedHeight =
        getValidResolution(imageScaleFactor, height, outputStride);
    const resizedWidth =
        getValidResolution(imageScaleFactor, width, outputStride);

    const {heatmapScores, offsets} = tf.tidy(() => {
      const inputTensor =
          toInputTensor(input, resizedHeight, resizedWidth, flipHorizontal);
      return this.predictForSinglePose(inputTensor, outputStride);
    });

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);

    heatmapScores.dispose();
    offsets.dispose();

    const scaleY = height / resizedHeight;
    const scaleX = width / resizedWidth;

    return scalePose(pose, scaleY, scaleX);
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
      input: InputType, imageScaleFactor = 0.5, flipHorizontal = false,
      outputStride: OutputStride = 16, maxDetections = 5, scoreThreshold = .5,
      nmsRadius = 20): Promise<Pose[]> {
    assertValidOutputStride(outputStride);
    assertValidScaleFactor(imageScaleFactor);

    const [height, width] = input instanceof tf.Tensor ?
        [input.shape[0], input.shape[1]] :
        [input.height, input.width];
    const resizedHeight =
        getValidResolution(imageScaleFactor, height, outputStride);
    const resizedWidth =
        getValidResolution(imageScaleFactor, width, outputStride);

    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        tf.tidy(() => {
          const inputTensor =
              toInputTensor(input, resizedHeight, resizedWidth, flipHorizontal);
          return this.predictForMultiPose(inputTensor, outputStride);
        });

    const poses = await decodeMultiplePoses(
        heatmapScores, offsets, displacementFwd, displacementBwd, outputStride,
        maxDetections, scoreThreshold, nmsRadius);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    const scaleY = height / resizedHeight;
    const scaleX = width / resizedWidth;

    return scalePoses(poses, scaleY, scaleX);
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

/**
 * Loads the PoseNet model instance from a checkpoint, with the MobileNet
 * architecture specified by the multiplier.
 *
 * @param multiplier An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. Defaults to 1.01. It is the float multiplier for the depth (number of
 * channels) for all convolution ops. The value corresponds to a MobileNet
 * architecture and checkpoint.  The larger the value, the larger the size of
 * the layers, and more accurate the model at the cost of speed.  Set this to a
 * smaller value to increase speed at the cost of accuracy.
 *
 */
export async function load(multiplier: MobileNetMultiplier = 1.01):
    Promise<PoseNet> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'number',
      `got multiplier type of ${typeof multiplier} when it should be a ` +
          `number.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier.toString()) >= 0,
      `invalid multiplier value of ${
          multiplier}.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  const mobileNet = await mobilenetLoader.load(multiplier);

  return new PoseNet(mobileNet);
}

export const mobilenetLoader = {
  load: async(multiplier: MobileNetMultiplier): Promise<MobileNet> => {
    const checkpoint = checkpoints[multiplier];

    const checkpointLoader = new CheckpointLoader(checkpoint.url);

    const variables = await checkpointLoader.getAllVariables();

    return new MobileNet(variables, checkpoint.architecture);
  }
};
