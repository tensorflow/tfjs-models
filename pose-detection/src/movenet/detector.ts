/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {MILLISECOND_TO_MICRO_SECONDS, SECOND_TO_MICRO_SECONDS} from '../calculators/constants';
import {getImageSize, toImageTensor} from '../calculators/image_utils';
import {ImageSize} from '../calculators/interfaces/common_interfaces';
import {BoundingBox} from '../calculators/interfaces/shape_interfaces';
import {isVideo} from '../calculators/is_video';
import {KeypointsOneEuroFilter} from '../calculators/keypoints_one_euro_filter';
import {LowPassFilter} from '../calculators/low_pass_filter';
import {COCO_KEYPOINTS} from '../constants';
import {PoseDetector} from '../pose_detector';
import {InputResolution, Pose, PoseDetectorInput, SupportedModels} from '../types';
import {getKeypointIndexByName} from '../util';

import {CROP_FILTER_ALPHA, KEYPOINT_FILTER_CONFIG, MIN_CROP_KEYPOINT_SCORE, MOVENET_CONFIG, MOVENET_ESTIMATION_CONFIG, MOVENET_MULTIPOSE_RESOLUTION, MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION, MOVENET_SINGLEPOSE_LIGHTNING_URL, MOVENET_SINGLEPOSE_THUNDER_RESOLUTION, MOVENET_SINGLEPOSE_THUNDER_URL, MULTIPOSE, MULTIPOSE_BOX_SCORE_IDX, MULTIPOSE_INSTANCE_SIZE, NUM_KEYPOINT_VALUES, NUM_KEYPOINTS, SINGLEPOSE_LIGHTNING, SINGLEPOSE_THUNDER} from './constants';
import {determineNextCropRegion, initCropRegion} from './crop_utils';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

/**
 * MoveNet detector class.
 */
class MoveNetDetector implements PoseDetector {
  private readonly modelInputResolution:
      InputResolution = {height: 0, width: 0};
  private readonly keypointIndexByName =
      getKeypointIndexByName(SupportedModels.MoveNet);
  private readonly multiPoseModel: boolean;
  private readonly enableSmoothing: boolean;

  // Global states.
  private readonly keypointsFilter =
      new KeypointsOneEuroFilter(KEYPOINT_FILTER_CONFIG);
  private readonly cropRegionFilterYMin = new LowPassFilter(CROP_FILTER_ALPHA);
  private readonly cropRegionFilterXMin = new LowPassFilter(CROP_FILTER_ALPHA);
  private readonly cropRegionFilterYMax = new LowPassFilter(CROP_FILTER_ALPHA);
  private readonly cropRegionFilterXMax = new LowPassFilter(CROP_FILTER_ALPHA);
  private cropRegion: BoundingBox;

  constructor(
      private readonly moveNetModel: tfc.GraphModel,
      config: MoveNetModelConfig) {
    // Only single-pose models have a fixed input resolution.
    if (config.modelType === SINGLEPOSE_LIGHTNING) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
      this.modelInputResolution.height =
          MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
    } else if (config.modelType === SINGLEPOSE_THUNDER) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
      this.modelInputResolution.height = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
    }
    this.multiPoseModel = config.modelType === MULTIPOSE;
    this.enableSmoothing = config.enableSmoothing;
  }

  /**
   * Runs inference on an image using a model that is assumed to be a single
   * person keypoint model that outputs 17 keypoints.
   *
   * @param inputImage 4D tensor containing the input image. Should be of size
   * [1, modelHeight, modelWidth, 3].
   * @return A `Pose`, or null if the model returned an unexpected tensor size.
   */
  async runSinglePersonPoseModel(inputImage: tf.Tensor4D): Promise<Pose|null> {
    const outputTensor = this.moveNetModel.execute(inputImage) as tf.Tensor;

    // We expect an output tensor of shape [1, 1, 17, 3] (batch, person,
    // keypoint, (y, x, score)).
    if (outputTensor.shape.length !== 4 || outputTensor.shape[0] !== 1 ||
        outputTensor.shape[1] !== 1 ||
        outputTensor.shape[2] !== NUM_KEYPOINTS ||
        outputTensor.shape[3] !== NUM_KEYPOINT_VALUES) {
      outputTensor.dispose();
      return null;
    }

    // Only use asynchronous downloads when we really have to (WebGPU) because
    // that will poll for download completion using setTimeOut which introduces
    // extra latency.
    let inferenceResult;
    if (tf.getBackend() !== 'webgpu') {
      inferenceResult = outputTensor.dataSync();
    } else {
      inferenceResult = await outputTensor.data();
    }
    outputTensor.dispose();

    const pose: Pose = {keypoints: [], score: 0.0};
    let numValidKeypoints = 0;
    for (let i = 0; i < NUM_KEYPOINTS; ++i) {
      pose.keypoints[i] = {
        y: inferenceResult[i * NUM_KEYPOINT_VALUES],
        x: inferenceResult[i * NUM_KEYPOINT_VALUES + 1],
        score: inferenceResult[i * NUM_KEYPOINT_VALUES + 2]
      };
      if (pose.keypoints[i].score > MIN_CROP_KEYPOINT_SCORE) {
        ++numValidKeypoints;
        pose.score += pose.keypoints[i].score;
      }
    }

    if (numValidKeypoints > 0) {
      pose.score /= numValidKeypoints;
    }

    return pose;
  }

  /**
   * Runs inference on an image using a model that is assumed to be a
   * multi-person keypoint model that outputs 17 keypoints and a box for a
   * multiple persons.
   *
   * @param inputImage 4D tensor containing the input image. Should be of size
   * [1, width, height, 3], where width and height are divisible by 32.
   * @return An array of an array of `Pose`s, or null if the model returned an
   * unexpected tensor size.
   */
  async runMultiPersonPoseModel(inputImage: tf.Tensor4D): Promise<Pose[]|null> {
    const outputTensor = this.moveNetModel.execute(inputImage) as tf.Tensor;

    // Multi-pose model output is a [1, n, 56] tensor ([batch, num_instances,
    // instance_keypoints_and_box]).
    if (outputTensor.shape.length !== 3 || outputTensor.shape[0] !== 1 ||
        outputTensor.shape[2] !== MULTIPOSE_INSTANCE_SIZE) {
      outputTensor.dispose();
      return null;
    }

    // Only use asynchronous downloads when we really have to (WebGPU) because
    // that will poll for download completion using setTimeOut which introduces
    // extra latency.
    let inferenceResult;
    if (tf.getBackend() !== 'webgpu') {
      inferenceResult = outputTensor.dataSync();
    } else {
      inferenceResult = await outputTensor.data();
    }
    outputTensor.dispose();

    const poses: Pose[] = [];

    const numInstances = inferenceResult.length / MULTIPOSE_INSTANCE_SIZE;
    for (let i = 0; i < numInstances; ++i) {
      poses[i] = {keypoints: []};
      const scoreIndex = i * MULTIPOSE_INSTANCE_SIZE + MULTIPOSE_BOX_SCORE_IDX;
      poses[i].score = inferenceResult[scoreIndex];
      poses[i].keypoints = [];
      for (let j = 0; j < NUM_KEYPOINTS; ++j) {
        poses[i].keypoints[j] = {
          y: inferenceResult
              [i * MULTIPOSE_INSTANCE_SIZE + j * NUM_KEYPOINT_VALUES],
          x: inferenceResult
              [i * MULTIPOSE_INSTANCE_SIZE + j * NUM_KEYPOINT_VALUES + 1],
          score: inferenceResult
              [i * MULTIPOSE_INSTANCE_SIZE + j * NUM_KEYPOINT_VALUES + 2]
        };
      }
    }

    return poses;
  }

  /**
   * Estimates poses for an image or video frame. This does standard ImageNet
   * pre-processing before inferring through the model. The image pixels should
   * have values [0-255]. It returns an array of poses.
   *
   * @param image ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement
   * The input image to feed through the network.
   * @param config Optional. Currently not used.
   * @param timestamp Optional. In milliseconds. This is useful when image is
   * a tensor, which doesn't have timestamp info. Or to override timestamp in a
   * video.
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig: MoveNetEstimationConfig = MOVENET_ESTIMATION_CONFIG,
      timestamp?: number): Promise<Pose[]> {
    estimationConfig = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    if (timestamp == null) {
      if (isVideo(image)) {
        timestamp = image.currentTime * SECOND_TO_MICRO_SECONDS;
      }
    } else {
      timestamp = timestamp * MILLISECOND_TO_MICRO_SECONDS;
    }

    const imageTensor3D = toImageTensor(image);
    const imageSize = getImageSize(imageTensor3D);
    const imageTensor4D: tf.Tensor4D = tf.expandDims(imageTensor3D, 0);

    // Make sure we don't dispose the input image if it's already a tensor.
    if (!(image instanceof tf.Tensor)) {
      imageTensor3D.dispose();
    }

    let poses: Pose[] = [];
    if (!this.multiPoseModel) {
      poses =
          [await this.estimateSinglePose(imageTensor4D, imageSize, timestamp)];
    } else {
      poses = await this.estimateMultiplePoses(imageTensor4D, imageSize);
    }

    // Convert keypoint coordinates from normalized coordinates to image space,
    // add keypoint names and calculate the overall pose score.
    for (let poseIdx = 0; poseIdx < poses.length; ++poseIdx) {
      for (let keypointIdx = 0; keypointIdx < poses[poseIdx].keypoints.length;
           ++keypointIdx) {
        poses[poseIdx].keypoints[keypointIdx].name =
            COCO_KEYPOINTS[keypointIdx];
        poses[poseIdx].keypoints[keypointIdx].y *= imageSize.height;
        poses[poseIdx].keypoints[keypointIdx].x *= imageSize.width;
      }
    }

    return poses;
  }

  /**
   * Runs a single-person keypoint model on an image, including the image
   * cropping and keypoint filtering logic.
   *
   * @param imageTensor4D A tf.Tensor4D that contains the input image.
   * @param imageSize: The width and height of the input image.
   * @param timestamp Image timestamp in milliseconds.
   * @return An array of `Keypoint` or null.
   */
  async estimateSinglePose(
      imageTensor4D: tf.Tensor4D, imageSize: ImageSize,
      timestamp: number): Promise<Pose|null> {
    if (!this.cropRegion) {
      this.cropRegion = initCropRegion(this.cropRegion == null, imageSize);
    }

    const croppedImage = tf.tidy(() => {
      // Crop region is a [batch, 4] size tensor.
      const cropRegionTensor = tf.tensor2d([[
        this.cropRegion.yMin, this.cropRegion.xMin, this.cropRegion.yMax,
        this.cropRegion.xMax
      ]]);
      // The batch index that the crop should operate on. A [batch] size
      // tensor.
      const boxInd: tf.Tensor1D = tf.zeros([1], 'int32');
      // Target size of each crop.
      const cropSize: [number, number] =
          [this.modelInputResolution.height, this.modelInputResolution.width];
      return tf.cast(
          tf.image.cropAndResize(
              imageTensor4D, cropRegionTensor, boxInd, cropSize, 'bilinear', 0),
          'int32');
    });
    imageTensor4D.dispose();

    const pose = await this.runSinglePersonPoseModel(croppedImage);
    croppedImage.dispose();

    if (pose == null || pose.score === 0.0) {
      this.reset();
      return {keypoints: [], score: 0.0};
    }

    // Convert keypoints from crop coordinates to image coordinates.
    for (let i = 0; i < pose.keypoints.length; ++i) {
      pose.keypoints[i].y =
          this.cropRegion.yMin + pose.keypoints[i].y * this.cropRegion.height;
      pose.keypoints[i].x =
          this.cropRegion.xMin + pose.keypoints[i].x * this.cropRegion.width;
    }

    // Apply the sequential filter before estimating the cropping area to make
    // it more stable.
    if (timestamp != null && this.enableSmoothing) {
      pose.keypoints = this.keypointsFilter.apply(
          pose.keypoints, timestamp, 1 /* objectScale */);
    }

    // Determine next crop region based on detected keypoints and if a crop
    // region is not detected, this will trigger the model to run on the full
    // image the next time estimatePoses() is called.
    const nextCropRegion = determineNextCropRegion(
        this.cropRegion, pose.keypoints, this.keypointIndexByName, imageSize);

    this.cropRegion = this.filterCropRegion(nextCropRegion);

    return pose;
  }

  /**
   * Runs a multi-person keypoint model on an image, including image
   * preprocessing.
   *
   * @param imageTensor4D A tf.Tensor4D that contains the input image.
   * @param imageSize: The width and height of the input image.
   * @return An array of `Keypoint` or null.
   */
  async estimateMultiplePoses(imageTensor4D: tf.Tensor4D, imageSize: ImageSize):
      Promise<Pose[]|null> {
    let resizedImage: tf.Tensor4D;
    let resizedWidth: number;
    let resizedHeight: number;
    let paddedImage: tf.Tensor4D;
    let paddedWidth: number;
    let paddedHeight: number;
    const dimensionDivisor = 32;  // Dimensions need to be divisible by 32.
    if (imageSize.width > imageSize.height) {
      resizedWidth = MOVENET_MULTIPOSE_RESOLUTION;
      resizedHeight = Math.round(
          MOVENET_MULTIPOSE_RESOLUTION * imageSize.height / imageSize.width);
      resizedImage =
          tf.image.resizeBilinear(imageTensor4D, [resizedHeight, resizedWidth]);

      paddedWidth = resizedWidth;
      paddedHeight =
          Math.ceil(resizedHeight / dimensionDivisor) * dimensionDivisor;
      paddedImage = tf.pad(
          resizedImage,
          [[0, 0], [0, paddedHeight - resizedHeight], [0, 0], [0, 0]]);
    } else {
      resizedWidth = Math.round(
          MOVENET_MULTIPOSE_RESOLUTION * imageSize.width / imageSize.height);
      resizedHeight = MOVENET_MULTIPOSE_RESOLUTION;
      resizedImage =
          tf.image.resizeBilinear(imageTensor4D, [resizedHeight, resizedWidth]);

      paddedWidth =
          Math.ceil(resizedWidth / dimensionDivisor) * dimensionDivisor;
      paddedHeight = resizedHeight;
      paddedImage = tf.pad(
          resizedImage,
          [[0, 0], [0, 0], [0, paddedWidth - resizedWidth], [0, 0]]);
    }
    resizedImage.dispose();
    imageTensor4D.dispose();

    const paddedImageInt32 = tf.cast(paddedImage, 'int32');
    paddedImage.dispose();
    const poses = await this.runMultiPersonPoseModel(paddedImageInt32);
    paddedImageInt32.dispose();

    // Convert keypoints from padded coordinates to normalized coordinates.
    for (let i = 0; i < poses.length; ++i) {
      for (let j = 0; j < poses[i].keypoints.length; ++j) {
        poses[i].keypoints[j].y =
            poses[i].keypoints[j].y * paddedHeight / resizedHeight;
        poses[i].keypoints[j].x =
            poses[i].keypoints[j].x * paddedWidth / resizedWidth;
      }
    }

    return poses;
  }

  filterCropRegion(newCropRegion: BoundingBox): BoundingBox {
    if (!newCropRegion) {
      this.cropRegionFilterYMin.reset();
      this.cropRegionFilterXMin.reset();
      this.cropRegionFilterYMax.reset();
      this.cropRegionFilterXMax.reset();
      return null;
    } else {
      const filteredYMin = this.cropRegionFilterYMin.apply(newCropRegion.yMin);
      const filteredXMin = this.cropRegionFilterXMin.apply(newCropRegion.xMin);
      const filteredYMax = this.cropRegionFilterYMax.apply(newCropRegion.yMax);
      const filteredXMax = this.cropRegionFilterXMax.apply(newCropRegion.xMax);
      return {
        yMin: filteredYMin,
        xMin: filteredXMin,
        yMax: filteredYMax,
        xMax: filteredXMax,
        height: filteredYMax - filteredYMin,
        width: filteredXMax - filteredXMin
      };
    }
  }

  dispose() {
    this.moveNetModel.dispose();
  }

  reset() {
    this.cropRegion = null;
    this.resetFilters();
  }

  resetFilters() {
    this.keypointsFilter.reset();
    this.cropRegionFilterYMin.reset();
    this.cropRegionFilterXMin.reset();
    this.cropRegionFilterYMax.reset();
    this.cropRegionFilterXMax.reset();
  }
}

/**
 * Loads the MoveNet model instance from a checkpoint. The model to be loaded
 * is configurable using the config dictionary `ModelConfig`. Please find more
 * details in the documentation of the `ModelConfig`.
 *
 * @param config `ModelConfig` dictionary that contains parameters for
 * the MoveNet loading process. Please find more details of each parameter
 * in the documentation of the `ModelConfig` interface.
 */
export async function load(modelConfig: MoveNetModelConfig = MOVENET_CONFIG):
    Promise<PoseDetector> {
  const config = validateModelConfig(modelConfig);
  let model: tfc.GraphModel;

  let fromTFHub = true;

  if (!!config.modelUrl) {
    fromTFHub = config.modelUrl.indexOf('https://tfhub.dev') > -1;
    model = await tfc.loadGraphModel(config.modelUrl, {fromTFHub});
  } else {
    let modelUrl;
    if (config.modelType === SINGLEPOSE_LIGHTNING) {
      modelUrl = MOVENET_SINGLEPOSE_LIGHTNING_URL;
    } else if (config.modelType === SINGLEPOSE_THUNDER) {
      modelUrl = MOVENET_SINGLEPOSE_THUNDER_URL;
    } else {
      throw new Error(`MoveNet multi-pose can only be loaded from a URL, ' +
        'not from TF.Hub yet.`);
    }
    model = await tfc.loadGraphModel(modelUrl, {fromTFHub});
  }
  return new MoveNetDetector(model, config);
}
