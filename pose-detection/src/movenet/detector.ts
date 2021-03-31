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

import { toImageTensor } from '../calculators/image_utils';
import { BasePoseDetector, PoseDetector } from '../pose_detector';
import { Keypoint, Pose, PoseDetectorInput } from '../types';

import { MOVENET_CONFIG, MOVENET_SINGLE_PERSON_ESTIMATION_CONFIG } from './constants';
import { validateEstimationConfig, validateModelConfig } from './detector_utils';
import { OneEuroFilter } from './one_euro_filter';
import { MoveNetEstimationConfig, MoveNetModelConfig } from './types';
import { KeypointModel } from './keypoint_model';

/**
 * MoveNet detector class.
 */
export class MoveNetDetector extends BasePoseDetector {
  private maxPoses: number;
  private model: KeypointModel;
  private modelWidth: number;
  private modelHeight: number;
  private cropRegion: number[];
  private filter: OneEuroFilter;
  private previousFrameTime: number;
  private frameTimeDiff: number;

  // Should not be called outside.
  private constructor(
    private readonly moveNetModel: KeypointModel,
    config: MoveNetModelConfig) {
    super();
    this.model = moveNetModel;
    // Should we retrieve these values from the config? Or the URL maybe?
    this.modelWidth = 192;
    this.modelHeight = 192;

    this.previousFrameTime = 0;
    // Assume 30 fps.
    this.frameTimeDiff = 0.0333;

    this.filter = new OneEuroFilter();
  }

  /**
   * Loads the MoveNet model instance from a checkpoint. The model to be loaded
   * is configurable using the config dictionary ModelConfig. Please find more
   * details in the documentation of the ModelConfig.
   *
   * @param config ModelConfig dictionary that contains parameters for
   * the MoveNet loading process. Please find more details of each parameters
   * in the documentation of the ModelConfig interface.
   */
  static async load(modelConfig: MoveNetModelConfig = MOVENET_CONFIG):
    Promise<PoseDetector> {
    const config = validateModelConfig(modelConfig);
    //    const defaultUrl = "";
    //    const model = await tfconv.loadGraphModel(config.modelUrl || defaultUrl);
    const model: KeypointModel = new KeypointModel();
    //    await model.load('http://localhost:8080/movenet/model.json');
    await model.load('http://localhost:8080/movenet/model.json');
    return new MoveNetDetector(model, config);
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It returns a
   * single pose or multiple poses based on the maxPose parameter from the
   * `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config
   *       maxPoses: Optional. Max number of poses to estimate.
   *       Only 1 pose is supported.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of `Pose`s.
   */
  async estimatePoses(
    image: PoseDetectorInput,
    estimationConfig:
      MoveNetEstimationConfig = MOVENET_SINGLE_PERSON_ESTIMATION_CONFIG):
    Promise<Pose[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      return [];
    }

    this.maxPoses = config.maxPoses;

    if (this.maxPoses > 1) {
      throw new Error('Multi-person poses is not implemented yet.');
    }

    // Keep track of fps for one euro filter.
    const now = performance.now();
    if (this.previousFrameTime !== 0) {
      this.frameTimeDiff = 0.02 * this.frameTimeDiff + 0.98 * (now - this.previousFrameTime);
    }
    this.previousFrameTime = now;

    const imageTensor3D = toImageTensor(image);
    const imageHeight = imageTensor3D.shape[0];
    const imageWidth = imageTensor3D.shape[1];
    const imageTensor4D = tf.expandDims(imageTensor3D, 0) as tf.Tensor4D;
    imageTensor3D.dispose();

    const [keypoints, ,] = await this.runInference(imageTensor4D, estimationConfig.minimumKeypointScore);

    for (let i = 0; i < keypoints.length; ++i) {
      keypoints[i].y *= imageHeight;
      keypoints[i].x *= imageWidth;
    }

    const poses: Pose[] = [];
    poses[0] = {
      'keypoints': keypoints
    };

    return poses;
  }

  determineCropRegion(keypoints: Keypoint[], webcamHeight: number, webcamWidth: number,
    minimumKeypointScore: number) {
    const keypointIndices: { [index: string]: number } = {
      nose: 0,
      left_eye: 1,
      right_eye: 2,
      left_ear: 3,
      right_ear: 4,
      left_shoulder: 5,
      right_shoulder: 6,
      left_elbow: 7,
      right_elbow: 8,
      left_wrist: 9,
      right_wrist: 10,
      left_hip: 11,
      right_hip: 12,
      left_knee: 13,
      right_knee: 14,
      left_ankle: 15,
      right_ankle: 16
    };

    const targetKeypoints: { [index: string]: number[] } = {
      nose: [0.0, 0.0],
      left_eye: [0.0, 0.0],
      right_eye: [0.0, 0.0],
      left_ear: [0.0, 0.0],
      right_ear: [0.0, 0.0],
      left_shoulder: [0.0, 0.0],
      right_shoulder: [0.0, 0.0],
      left_elbow: [0.0, 0.0],
      right_elbow: [0.0, 0.0],
      left_wrist: [0.0, 0.0],
      right_wrist: [0.0, 0.0],
      left_hip: [0.0, 0.0],
      right_hip: [0.0, 0.0],
      left_knee: [0.0, 0.0],
      right_knee: [0.0, 0.0],
      left_ankle: [0.0, 0.0],
      right_ankle: [0.0, 0.0],
    };

    for (const key of Object.keys(targetKeypoints)) {
      targetKeypoints[key] = [
        keypoints[keypointIndices[key]].y * webcamHeight,
        keypoints[keypointIndices[key]].x * webcamWidth
      ];
    }

    if (keypoints[keypointIndices['left_hip']].score > minimumKeypointScore &&
      keypoints[keypointIndices['right_hip']].score > minimumKeypointScore &&
      keypoints[keypointIndices['left_shoulder']].score > minimumKeypointScore &&
      keypoints[keypointIndices['right_shoulder']].score > minimumKeypointScore) {
      let centerX = 0.0;
      let centerY = 0.0;
      centerY =
        (targetKeypoints['left_hip'][0] + targetKeypoints['right_hip'][0]) / 2;
      centerX =
        (targetKeypoints['left_hip'][1] + targetKeypoints['right_hip'][1]) / 2;

      const torsoJoints = [
        'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'];
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
        if (keypoints[keypointIndices[key]].score < minimumKeypointScore) {
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

      let cropLengthHalf = Math.max(
        maxTorsoXrange * 2.0,
        maxTorsoYrange * 2.0,
        maxBodyYrange * 1.2,
        maxBodyXrange * 1.2);

      cropLengthHalf = Math.min(
        cropLengthHalf,
        Math.max(centerX, webcamWidth - centerX, centerY, webcamHeight - centerY)
      );

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

  async runInference(inputImage: tf.Tensor4D, minimumKeypointScore: number):
    Promise<[Keypoint[], number, number[]]> {
    let keypoints: Keypoint[] = null;
    let processingTime = 0.0;
    let visualizeCropRegion = null;

    if (this.cropRegion != null) {
      const croppedImage = tf.tidy(() => {
        // Crop region is a [batch, 4] size tensor.
        const cropRegionTensor = tf.tensor2d([this.cropRegion]);
        // The batch index that the crop should operate on. A [batch] size tensor.
        const boxInd = tf.zeros([1], 'int32') as tf.Tensor1D;
        // Target size of each crop.
        const cropSize: [number, number] = [this.modelHeight, this.modelWidth];
        return tf.cast(tf.image.cropAndResize(inputImage, cropRegionTensor, boxInd, cropSize, 'bilinear', 0), 'int32');
      });

      // Run cropModel. Model will dispose croppedImage.
      keypoints = await this.model.detectKeypoints(croppedImage);

      processingTime += this.model.getProcessingTime();

      // Convert keypoints to image coordinates.
      const cropHeight = this.cropRegion[2] - this.cropRegion[0];
      const cropWidth = this.cropRegion[3] - this.cropRegion[1];
      for (let i = 0; i < keypoints.length; ++i) {
        keypoints[i].y = this.cropRegion[0] + keypoints[i].y * cropHeight;
        keypoints[i].x = this.cropRegion[1] + keypoints[i].x * cropWidth;
      }

      // Apply the sequential filter before estimating the cropping area
      // to make it more stable.
      this.arrayToKeypoints(this.filter.insert(this.keypointsToArray(keypoints), 1.0 / this.frameTimeDiff), keypoints);

      // Determine next crop region based on detected keypoints and if a crop
      // region is not detected, this will trigger the full model to run.
      visualizeCropRegion = this.cropRegion;

      // Use exponential filter on the cropping region to make it less jittery.
      let newCropRegion = this.determineCropRegion(keypoints, inputImage.shape[1], inputImage.shape[2], minimumKeypointScore);
      if (newCropRegion != null) {
        newCropRegion = newCropRegion.map(x => x * 0.9);
        this.cropRegion = this.cropRegion.map(x => x * 0.1);
        this.cropRegion = this.cropRegion.map((e, i) => e + newCropRegion[i]);
      } else {
        this.cropRegion = null;
      }
    }

    if (!this.cropRegion) {
      const resizedImage: tf.Tensor = tf.image.resizeBilinear(inputImage, [this.modelHeight, this.modelWidth]);
      const resizedImageInt = tf.cast(resizedImage, 'int32');
      resizedImage.dispose();

      // Model will dispose resizedImageInt.
      keypoints = await this.model.detectKeypoints(resizedImageInt, true);
      processingTime += this.model.getProcessingTime();

      this.arrayToKeypoints(this.filter.insert(this.keypointsToArray(keypoints), 1.0 / this.frameTimeDiff), keypoints);

      // Determine crop region based on detected keypoints.
      visualizeCropRegion = null;
      this.cropRegion = this.determineCropRegion(keypoints, inputImage.shape[1], inputImage.shape[2], minimumKeypointScore);
    }

    inputImage.dispose();

    return [keypoints, processingTime, visualizeCropRegion];
  }

  dispose() {
    this.moveNetModel.dispose();
  }
}
