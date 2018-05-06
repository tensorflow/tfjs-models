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

import * as tf from '@tensorflow/tfjs-core';

import {CheckpointLoader} from './checkpoint_loader';
import {checkpoints} from './checkpoints';
import {assertValidOutputStride, MobileNet, OutputStride} from './mobilenet';
import decodeMultiplePoses from './multiPose/decodeMultiplePoses';
import decodeSinglePose from './singlePose/decodeSinglePose';
import {Pose} from './types';
import {scalePose, scalePoses} from './util';

export type InputType =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement;

function toInputTensor(
    input: InputType, inputSize: number, reverse: boolean): tf.Tensor3D {
  const imageTensor = tf.fromPixels(input);

  if (reverse)
    return imageTensor.reverse(1).resizeBilinear([inputSize, inputSize]);
  else
    return imageTensor.resizeBilinear([inputSize, inputSize]);
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
    assertValidOutputStride(outputStride);
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
   * @param inputSize The size the input should be resized to before feeding
   * it through the network.  Defaults to 513.  Must have a value which when
   * 1 is subtracted from it, is divisible by the output stride. Sample
   * acceptable values are 29, 161, 193, 257, 289, 321, 353, 385, 417, 449, 481,
   * 513. Set this number lower to scale down the image and increase the speed
   * when feeding through the network.
   *
   * @param reverse.  A boolean which defaults to false.  If set to true,
   * reverses the image horizontally before feeding through the network.  Useful
   * for videos where the image is often reversed horizontally.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return A single pose with a confidence score, which contains an array of
   * keypoints indexed by part id, each with a score and position.
   */
  async estimateSinglePose(
      input: InputType, inputSize: number = 513, reverse: boolean = false,
      outputStride: OutputStride = 16): Promise<Pose> {
    const {heatmapScores, offsets} = tf.tidy(() => {
      const inputTensor = toInputTensor(input, inputSize, reverse);
      return this.predictForSinglePose(inputTensor, outputStride);
    })

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);

    heatmapScores.dispose();
    offsets.dispose();

    const scale = input.width / inputSize;

    return scalePose(pose, scale);
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
   * @param inputSize The size the input should be resized to before feeding
   * it through the network.  Defaults to 513.  Must have a value which when
   * 1 is subtracted from it, is divisible by the output stride. Sample
   * acceptable values are 29, 161, 193, 257, 289, 321, 353, 385, 417, 449, 481,
   * 513. Set this number lower to scale down the image and increase the speed
   * when feeding through the network.
   *
   * @param reverse.  A boolean which defaults to false.  If set to true,
   * reverses the image horizontally before feeding through the network.  Useful
   * for videos where the image is often reversed horizontally.
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
   * the corresponding keypoint scores.
   */
  async estimateMultiplePoses(
      input: InputType, inputSize: number = 513, reverse: boolean = false,
      outputStride: OutputStride = 16, maxDetections = 5, scoreThreshold = .5,
      nmsRadius = 20): Promise<Pose[]> {
    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        tf.tidy(() => {
          const inputTensor = toInputTensor(input, inputSize, reverse);
          return this.predictForMultiPose(inputTensor, outputStride);
        })

    const poses = await decodeMultiplePoses(
        heatmapScores, offsets, displacementFwd, displacementBwd, outputStride,
        maxDetections, scoreThreshold, nmsRadius);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    const scale = input.width / inputSize;

    return scalePoses(poses, scale);
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

/**
 * Loads the PoseNet model instance from a checkpoint, with the MobileNet
 * architecture specified by the multiplier.
 *
 * @param multiplier An optional string with values: "1.01", "1.00, ""0.75", or
 * "0.50". Defaults to "1.01". It is the string with the value of the float
 * multiplier for the depth (number of channels) for all convolution ops. The
 * value corresponds to an MobileNet architecture and checkpoint.  The larger
 * the value, the larger the size of the layers, and more accurate the model at
 * the cost of speed.  Set this to a smaller value to increase speed at the cost
 * of accuracy.
 *
 * @return
 */
export default async function load(multiplier: string = '1.01'):
    Promise<PoseNet> {
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'string',
      `Error: got multiplier type of ${
          typeof multiplier} when it should be a ` +
          `string.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier) >= 0,
      `Error: Invalid multiplier value of ${
          multiplier}.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  // get the checkpoint for the multiplier
  const checkpoint = checkpoints[multiplier];

  const checkpointLoader = new CheckpointLoader(checkpoint.url);

  const variables = await checkpointLoader.getAllVariables();

  const mobileNet = new MobileNet(variables, checkpoint.architecture);

  return new PoseNet(mobileNet);
}
