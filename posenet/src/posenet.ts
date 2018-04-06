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
import {CheckpointLoader} from './deeplearn-legacy-loader';

import * as multiPose from './multiPose';
import * as singlePose from './singlePose';
import {Pose} from './types';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/cl-move-mirror.appspot.com/';

export type OutputStride = 32|16|8;

export class PoseNet {
  private variables: {[varName: string]: tf.Tensor};

  // yolo variables
  private PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
  private ONE = tf.scalar(1);

  /**
   * Loads necessary variables for PoseNet.
   */
  async load(): Promise<void> {
    const checkpointLoader = new CheckpointLoader(
        GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_val2017_232/');
    this.variables = await checkpointLoader.getAllVariables();
  }

  private mobileNet(input: tf.Tensor3D, outputStride: OutputStride) {
    // Normalize the pixels [0, 255] to be between [-1, 1].
    const preprocessedInput =
        input.div(this.PREPROCESS_DIVISOR).sub(this.ONE) as tf.Tensor3D;

    const conv2d0Stride = 2;

    const x1 = preprocessedInput
                   .conv2d(this.weights('Conv2d_0'), conv2d0Stride, 'same')
                   .add(this.biases('Conv2d_0'))
                   // relu6
                   .clipByValue(0, 6) as tf.Tensor3D;

    let previousLayer = x1;

    const convStridesAndLayerNames: number[][] = [
      [1, 1], [2, 2], [1, 3], [2, 4], [1, 5], [2, 6], [1, 7], [1, 8], [1, 9],
      [1, 10], [1, 11], [2, 12], [1, 13]
    ];

    // The currentStride variable keeps track of the output stride of
    // the activations, i.e., the running product of convolution
    // strides up to the current network layer. This allows us to
    // invoke atrous convolution whenever applying the next
    // convolution would result in the activations having output
    // stride larger than the target outputStride.
    let currentStride = conv2d0Stride;

    // The atrous convolution rate parameter.
    let rate = 1;

    convStridesAndLayerNames.forEach(([stride, layerName]) => {
      let layerStride, layerRate;

      if (currentStride === outputStride) {
        // If we have reached the target outputStride, then we need to
        // employ atrous convolution with stride=1 and multiply the atrous
        // rate by the current unit's stride for use in subsequent layers.
        layerStride = 1;
        layerRate = rate;
        rate *= stride;
      } else {
        layerStride = stride;
        layerRate = 1;
        currentStride *= stride;
      }

      const layer = this.depthwiseConvBlock(
          previousLayer, layerStride, layerName, layerRate);
      previousLayer = layer;
    });

    return previousLayer;
  }

  /**
   * Infer through PoseNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns the heatmaps and offsets.  Infers through the outputs
   * that are needed for single pose decoding
   *
   * @param input un-preprocessed input image.
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return heatmapScores, offsets
   */
  predictForSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 32):
      {heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D} {
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet(input, outputStride);

      const heatmaps =
          mobileNetOutput.conv2d(this.weights('heatmap_2'), 1, 'same')
              .add(this.biases('heatmap_2')) as tf.Tensor3D;

      const offsets =
          mobileNetOutput.conv2d(this.weights('offset_2'), 1, 'same')
              .add(this.biases('offset_2')) as tf.Tensor3D;

      return {heatmapScores: heatmaps.sigmoid(), offsets};
    });
  }

  /**
   * Infer through PoseNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns the heatmaps and offsets.  Infers through the outputs
   * that are needed for single pose decoding
   *
   * @param input un-preprocessed input image.
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
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet(input, outputStride);

      const heatmaps =
          mobileNetOutput.conv2d(this.weights('heatmap_2'), 1, 'same')
              .add(this.biases('heatmap_2')) as tf.Tensor3D;

      const offsets =
          mobileNetOutput.conv2d(this.weights('offset_2'), 1, 'same')
              .add(this.biases('offset_2')) as tf.Tensor3D;

      const displacementFwd =
          mobileNetOutput.conv2d(this.weights('displacement_fwd_2'), 1, 'same')
              .add(this.biases('displacement_fwd_2')) as tf.Tensor3D;

      const displacementBwd =
          mobileNetOutput.conv2d(this.weights('displacement_bwd_2'), 1, 'same')
              .add(this.biases('displacement_bwd_2')) as tf.Tensor3D;

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
   * pre-processing before inferring through the model. This
   * method returns a single pose.
   *
   * @param input un-preprocessed input image.
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   * @return A single pose with a confidence score, which contains an array of
   * keypoints indexed by part id, each with a score and position.
   */
  estimateSinglePose(input: tf.Tensor3D, outputStride: OutputStride = 32):
      Pose {
    const {heatmapScores, offsets} =
        this.predictForSinglePose(input, outputStride);

    const pose = singlePose.decode(heatmapScores, offsets, outputStride);

    heatmapScores.dispose();
    offsets.dispose();

    return pose;
  }

  /**
   * Infer through PoseNet, and estimates multiple poses using the outputs.
   * assumes variables have been loaded. This does standard ImageNet
   * pre-processing before inferring through the model.  It detects
   * multiple poses and finds their parts from part scores and
   * displacement vectors using a fast greedy decoding algorithm.  It returns
   * up to `maxDetections` object instance detections in decreasing root score
   * order.
   *
   * @param input un-preprocessed input image.
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

    const poses = await multiPose.decode(
        heatmapScores, offsets, displacementFwd, displacementBwd, outputStride,
        maxDetections, scoreThreshold, nmsRadius);

    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();

    return poses;
  }

  private depthwiseConvBlock(
      inputs: tf.Tensor3D, stride: number, blockID: number, dilations = 1) {
    const dwLayer = `Conv2d_${String(blockID)}_depthwise`;
    const pwLayer = `Conv2d_${String(blockID)}_pointwise`;

    const x1 = inputs
                   .depthwiseConv2D(
                       this.depthwiseWeights(dwLayer), stride, 'same', 'NHWC',
                       dilations)
                   .add(this.biases(dwLayer))
                   // relu6
                   .clipByValue(0, 6) as tf.Tensor3D;

    const x2 = x1.conv2d(this.weights(pwLayer), [1, 1], 'same')
                   .add(this.biases(pwLayer))
                   // relu6
                   .clipByValue(0, 6) as tf.Tensor3D;

    return x2;
  }

  private weights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/weights`] as tf.Tensor4D;
  }

  private biases(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/biases`] as tf.Tensor1D;
  }

  private depthwiseWeights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/depthwise_weights`] as
        tf.Tensor4D;
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
