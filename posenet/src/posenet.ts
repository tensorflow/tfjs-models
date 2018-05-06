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
import {assertValidOutputStride, ConvolutionDefinition, MobileNet, OutputStride} from './mobilenet';
import decodeMultiplePoses from './multiPose/decodeMultiplePoses';
import decodeSinglePose from './singlePose/decodeSinglePose';
import {Pose} from './types';

const defaultCheckpoint = checkpoints[101];

export class PoseNet {
  mobileNet: MobileNet;

  constructor(mobileNet: MobileNet) {
    this.mobileNet = mobileNet;
  }

  /**
   * Infer through PoseNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model.
   * The image should pixels should have values [0-255]. This
   * method returns the heatmaps and offsets.  Infers through the outputs
   * that are needed for single pose decoding
   *
   * @param input un-preprocessed input image, with values in range [0-255]
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return heatmapScores, offsets
   */
  predictForSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 32):
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
   * Infer through PoseNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model.
   * The image should pixels should have values [0-255]. Infers
   * through the outputs that are needed for multiple pose decoding. This
   * method returns the heatmaps offsets, and mid-range displacements.
   *
   * @param input un-preprocessed input image, with values in range [0-255]
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return heatmapScores, offsets, displacementFwd, displacementBwd
   */
  predictForMultiPose(input: tf.Tensor3D, outputStride: OutputStride = 32): {
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
   * Infer through PoseNet, and estimates a single pose using the outputs.
   * assumes variables have been loaded. This does standard ImageNet
   * pre-processing before inferring through the model.
   * The image should pixels should have values [0-255].
   * This method returns a single pose.
   *
   * @param input un-preprocessed input image, with values in range [0-255].
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return A single pose with a confidence score, which contains an array of
   * keypoints indexed by part id, each with a score and position.
   */
  async estimateSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 32):
      Promise<Pose> {
    const {heatmapScores, offsets} =
        this.predictForSinglePose(input, outputStride);

    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);

    heatmapScores.dispose();
    offsets.dispose();

    return pose;
  }

  /**
   * Infer through PoseNet, and estimates multiple poses using the outputs.
   * assumes variables have been loaded. This does standard ImageNet
   * pre-processing before inferring through the model.
   * The image should pixels should have values [0-255].
   * It detects multiple poses and finds their parts from part scores and
   * displacement vectors using a fast greedy decoding algorithm.  It returns
   * up to `maxDetections` object instance detections in decreasing root score
   * order.
   *
   * @param input un-preprocessed input image, with values in range [0-255].
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
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
      input: tf.Tensor3D, outputStride: OutputStride = 32, maxDetections = 5,
      scoreThreshold = .5, nmsRadius = 20): Promise<Pose[]> {
    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        this.predictForMultiPose(input, outputStride);

    const poses = await decodeMultiplePoses(
        heatmapScores, offsets, displacementFwd, displacementBwd, outputStride,
        maxDetections, scoreThreshold, nmsRadius);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return poses;
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

export default async function posenet(
    checkpointUrl: string = defaultCheckpoint.url,
    convolutionDefinitions: ConvolutionDefinition[] =
        defaultCheckpoint.architecture): Promise<PoseNet> {
  const checkpointLoader = new CheckpointLoader(checkpointUrl);

  const variables = await checkpointLoader.getAllVariables();

  const mobileNet = new MobileNet(variables, convolutionDefinitions);

  return new PoseNet(mobileNet);
}
