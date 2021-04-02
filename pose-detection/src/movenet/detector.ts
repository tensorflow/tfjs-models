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

import * as tf from '@tensorflow/tfjs-core';

import {toImageTensor} from '../calculators/image_utils';
import {BasePoseDetector, PoseDetector} from '../pose_detector';
import {InputResolution, Keypoint, Pose, PoseDetectorInput} from '../types';

import {ARCHITECTURE_SINGLEPOSE_LIGHTNING, ARCHITECTURE_SINGLEPOSE_THUNDER, KPT_INDICES, MIN_CROP_KEYPOINT_SCORE, MOVENET_CONFIG, MOVENET_SINGLE_POSE_ESTIMATION_CONFIG, MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION, MOVENET_SINGLEPOSE_LIGHTNING_URL, MOVENET_SINGLEPOSE_THUNDER_RESOLUTION, MOVENET_SINGLEPOSE_THUNDER_URL} from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {KeypointModel} from './keypoint_model';
import {RobustOneEuroFilter} from './robust_one_euro_filter';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

/**
 * MoveNet detector class.
 */
export class MoveNetDetector extends BasePoseDetector {
  private model: KeypointModel;
  private modelInputResolution: InputResolution = {height: 0, width: 0};
  private cropRegion: number[];
  private filter: RobustOneEuroFilter;
  private previousFrameTime: number;
  private frameTimeDiff: number;

  // Should not be called outside.
  private constructor(
      private readonly moveNetModel: KeypointModel,
      config: MoveNetModelConfig) {
    super();
    this.model = moveNetModel;

    if (config.modelType === ARCHITECTURE_SINGLEPOSE_LIGHTNING) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
      this.modelInputResolution.height =
          MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
    } else if (config.modelType === ARCHITECTURE_SINGLEPOSE_THUNDER) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
      this.modelInputResolution.height = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
    }

    this.previousFrameTime = 0;
    // This will be used to calculate the actual camera fps. Starts with 30 fps
    // as an assumption.
    this.frameTimeDiff = 0.0333;

    this.filter = new RobustOneEuroFilter();
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
  static async load(modelConfig: MoveNetModelConfig = MOVENET_CONFIG):
      Promise<PoseDetector> {
    const config = validateModelConfig(modelConfig);
    const model: KeypointModel = new KeypointModel();
    if (config.modelUrl) {
      await model.load(config.modelUrl);
    } else {
      let modelUrl;
      if (config.modelType === ARCHITECTURE_SINGLEPOSE_LIGHTNING) {
        modelUrl = MOVENET_SINGLEPOSE_LIGHTNING_URL;
      } else if (config.modelType === ARCHITECTURE_SINGLEPOSE_THUNDER) {
        modelUrl = MOVENET_SINGLEPOSE_THUNDER_URL;
      }
      await model.load(modelUrl, true);
    }
    return new MoveNetDetector(model, config);
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It returns a
   * single pose.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config
   *
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig:
          MoveNetEstimationConfig = MOVENET_SINGLE_POSE_ESTIMATION_CONFIG):
      Promise<Pose[]> {
    // We only validate that maxPoses is 1.
    validateEstimationConfig(estimationConfig);

    if (image == null) {
      return [];
    }

    // Keep track of fps for one euro filter. The 'window' check is to make sure
    // the detector can run in NodeJS, that doesn't have performance.now()
    // without require('perf_hooks').
    if (typeof window !== 'undefined') {
      const now = performance.now();
      if (this.previousFrameTime !== 0) {
        const newSampleWeight = 0.02;
        this.frameTimeDiff = (1.0 - newSampleWeight) * this.frameTimeDiff +
            newSampleWeight * (now - this.previousFrameTime);
      }
      this.previousFrameTime = now;
    }

    const imageTensor3D = toImageTensor(image);
    const imageHeight = imageTensor3D.shape[0];
    const imageWidth = imageTensor3D.shape[1];
    const imageTensor4D: tf.Tensor4D = tf.expandDims(imageTensor3D, 0);

    // Make sure we don't dispose the input image if it's already a tensor.
    if (!(image instanceof tf.Tensor)) {
      imageTensor3D.dispose();
    }

    let keypoints: Keypoint[] = null;

    // If we have a cropRegion from a previous run, try to run the model on an
    // image crop first.
    if (this.cropRegion != null) {
      const croppedImage = tf.tidy(() => {
        // Crop region is a [batch, 4] size tensor.
        const cropRegionTensor = tf.tensor2d([this.cropRegion]);
        // The batch index that the crop should operate on. A [batch] size
        // tensor.
        const boxInd = tf.zeros([1], 'int32') as tf.Tensor1D;
        // Target size of each crop.
        const cropSize: [number, number] =
            [this.modelInputResolution.height, this.modelInputResolution.width];
        return tf.cast(
            tf.image.cropAndResize(
                imageTensor4D, cropRegionTensor, boxInd, cropSize, 'bilinear',
                0),
            'int32');
      });

      // Run cropModel. Model will dispose croppedImage.
      keypoints = await this.model.detectKeypoints(croppedImage);

      // Convert keypoints to image coordinates.
      const cropHeight = this.cropRegion[2] - this.cropRegion[0];
      const cropWidth = this.cropRegion[3] - this.cropRegion[1];
      for (let i = 0; i < keypoints.length; ++i) {
        keypoints[i].y = this.cropRegion[0] + keypoints[i].y * cropHeight;
        keypoints[i].x = this.cropRegion[1] + keypoints[i].x * cropWidth;
      }

      // Apply the sequential filter before estimating the cropping area
      // to make it more stable.
      this.arrayToKeypoints(
          this.filter.insert(
              this.keypointsToArray(keypoints), 1.0 / this.frameTimeDiff),
          keypoints);

      // Determine next crop region based on detected keypoints and if a crop
      // region is not detected, this will trigger the model to run on the full
      // image.
      let newCropRegion = this.determineCropRegion(
          keypoints, imageTensor4D.shape[1], imageTensor4D.shape[2]);

      // Use exponential filter on the cropping region to make it less jittery.
      if (newCropRegion != null) {
        newCropRegion = newCropRegion.map(x => x * 0.9);
        this.cropRegion = this.cropRegion.map(x => x * 0.1);
        this.cropRegion = this.cropRegion.map((e, i) => e + newCropRegion[i]);
      } else {
        this.cropRegion = null;
      }
    } else {
      // No cropRegion was available from a previous run, so run the model on
      // the full image.
      const resizedImage: tf.Tensor = tf.image.resizeBilinear(
          imageTensor4D,
          [this.modelInputResolution.height, this.modelInputResolution.width]);
      const resizedImageInt = tf.cast(resizedImage, 'int32') as tf.Tensor4D;
      resizedImage.dispose();

      // Model will dispose resizedImageInt.
      keypoints = await this.model.detectKeypoints(resizedImageInt, true);

      this.arrayToKeypoints(
          this.filter.insert(
              this.keypointsToArray(keypoints), 1.0 / this.frameTimeDiff),
          keypoints);

      // Determine crop region based on detected keypoints.
      this.cropRegion = this.determineCropRegion(
          keypoints, imageTensor4D.shape[1], imageTensor4D.shape[2]);
    }

    imageTensor4D.dispose();

    // Convert keypoint coordinates from normalized coordinates to image space
    // and leave out keypoints that don't have a high enough confidence.
    for (let i = 0; i < keypoints.length; ++i) {
      keypoints[i].y *= imageHeight;
      keypoints[i].x *= imageWidth;
    }

    const poses: Pose[] = [];
    poses[0] = {'keypoints': keypoints};

    return poses;
  }

  torsoVisible(keypoints: Keypoint[]) {
    return (
        keypoints[KPT_INDICES['left_hip']].score > MIN_CROP_KEYPOINT_SCORE &&
        keypoints[KPT_INDICES['right_hip']].score > MIN_CROP_KEYPOINT_SCORE &&
        keypoints[KPT_INDICES['left_shoulder']].score >
            MIN_CROP_KEYPOINT_SCORE &&
        keypoints[KPT_INDICES['right_shoulder']].score >
            MIN_CROP_KEYPOINT_SCORE);
  }

  determineTorsoAndBodyRange(
      keypoints: Keypoint[], targetKeypoints: {[index: string]: number[]},
      centerY: number, centerX: number) {
    const torsoJoints =
        ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'];
    let maxTorsoYrange = 0.0;
    let maxTorsoXrange = 0.0;
    for (let i = 0; i < torsoJoints.length; i++) {
      const distY = Math.abs(centerY - targetKeypoints[torsoJoints[i]][0]);
      const distX = Math.abs(centerX - targetKeypoints[torsoJoints[i]][1]);
      if (distY > maxTorsoYrange) {
        maxTorsoYrange = distY;
      }
      if (distX > maxTorsoXrange) {
        maxTorsoXrange = distX;
      }
    }
    let maxBodyYrange = 0.0;
    let maxBodyXrange = 0.0;
    for (const key of Object.keys(targetKeypoints)) {
      if (keypoints[KPT_INDICES[key]].score < MIN_CROP_KEYPOINT_SCORE) {
        continue;
      }
      const distY = Math.abs(centerY - targetKeypoints[key][0]);
      const distX = Math.abs(centerX - targetKeypoints[key][1]);
      if (distY > maxBodyYrange) {
        maxBodyYrange = distY;
      }
      if (distX > maxBodyXrange) {
        maxBodyXrange = distX;
      }
    }

    return [maxTorsoYrange, maxTorsoXrange, maxBodyYrange, maxBodyXrange];
  }

  determineCropRegion(
      keypoints: Keypoint[], webcamHeight: number, webcamWidth: number) {
    const targetKeypoints: {[index: string]: number[]} = {};

    for (const key of Object.keys(targetKeypoints)) {
      targetKeypoints[key] = [
        keypoints[KPT_INDICES[key]].y * webcamHeight,
        keypoints[KPT_INDICES[key]].x * webcamWidth
      ];
    }

    if (this.torsoVisible(keypoints)) {
      const centerY =
          (targetKeypoints['left_hip'][0] + targetKeypoints['right_hip'][0]) /
          2;
      const centerX =
          (targetKeypoints['left_hip'][1] + targetKeypoints['right_hip'][1]) /
          2;

      const [maxTorsoYrange, maxTorsoXrange, maxBodyYrange, maxBodyXrange] =
          this.determineTorsoAndBodyRange(
              keypoints, targetKeypoints, centerY, centerX);

      let cropLengthHalf = Math.max(
          maxTorsoXrange * 2.0, maxTorsoYrange * 2.0, maxBodyYrange * 1.2,
          maxBodyXrange * 1.2);

      cropLengthHalf = Math.min(
          cropLengthHalf,
          Math.max(
              centerX, webcamWidth - centerX, centerY, webcamHeight - centerY));

      let cropCorner = [centerY - cropLengthHalf, centerX - cropLengthHalf];

      if (cropLengthHalf > Math.max(webcamWidth, webcamHeight) / 2) {
        cropLengthHalf = Math.max(webcamWidth, webcamHeight) / 2;
        cropCorner = [0.0, 0.0];
      }

      const cropLength = cropLengthHalf * 2;
      const cropRegion = [
        cropCorner[0] / webcamHeight, cropCorner[1] / webcamWidth,
        (cropCorner[0] + cropLength) / webcamHeight,
        (cropCorner[1] + cropLength) / webcamWidth
      ];
      return cropRegion;
    } else {
      return null;
    }
  }

  keypointsToArray(keypoints: Keypoint[]) {
    const values: number[] = [];
    for (let i = 0; i < 17; ++i) {
      values[i * 2] = keypoints[i].y;
      values[i * 2 + 1] = keypoints[i].x;
    }
    return values;
  }

  arrayToKeypoints(values: number[], keypoints: Keypoint[]) {
    for (let i = 0; i < 17; ++i) {
      keypoints[i].y = values[i * 2];
      keypoints[i].x = values[i * 2 + 1];
    }
  }

  dispose() {
    this.moveNetModel.dispose();
  }
}
