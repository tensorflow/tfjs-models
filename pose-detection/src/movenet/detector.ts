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

import {SECOND_TO_MICRO_SECONDS} from '../calculators/constants';
import {getImageSize, toImageTensor} from '../calculators/image_utils';
import {BoundingBox} from '../calculators/interfaces/shape_interfaces';
import {isVideo} from '../calculators/is_video';
import {KeypointsOneEuroFilter} from '../calculators/keypoints_one_euro_filter';
import {LowPassFilter} from '../calculators/low_pass_filter';
import {COCO_KEYPOINTS_NAMED_MAP} from '../constants';
import {BasePoseDetector, PoseDetector} from '../pose_detector';
import {InputResolution, Keypoint, Pose, PoseDetectorInput} from '../types';

import {CROP_FILTER_ALPHA, KEYPOINT_FILTER_CONFIG, MIN_CROP_KEYPOINT_SCORE, MOVENET_CONFIG, MOVENET_SINGLE_POSE_ESTIMATION_CONFIG, MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION, MOVENET_SINGLEPOSE_LIGHTNING_URL, MOVENET_SINGLEPOSE_THUNDER_RESOLUTION, MOVENET_SINGLEPOSE_THUNDER_URL, SINGLEPOSE_LIGHTNING, SINGLEPOSE_THUNDER} from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

/**
 * MoveNet detector class.
 */
export class MoveNetDetector extends BasePoseDetector {
  private readonly modelInputResolution:
      InputResolution = {height: 0, width: 0};
  // Global states.
  private keypointsFilter = new KeypointsOneEuroFilter(KEYPOINT_FILTER_CONFIG);
  private cropRegion: BoundingBox;
  private cropRegionFilterYMin = new LowPassFilter(CROP_FILTER_ALPHA);
  private cropRegionFilterXMin = new LowPassFilter(CROP_FILTER_ALPHA);
  private cropRegionFilterYMax = new LowPassFilter(CROP_FILTER_ALPHA);
  private cropRegionFilterXMax = new LowPassFilter(CROP_FILTER_ALPHA);

  // Should not be called outside.
  private constructor(
      private readonly moveNetModel: tfc.GraphModel,
      config: MoveNetModelConfig) {
    super();

    if (config.modelType === SINGLEPOSE_LIGHTNING) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
      this.modelInputResolution.height =
          MOVENET_SINGLEPOSE_LIGHTNING_RESOLUTION;
    } else if (config.modelType === SINGLEPOSE_THUNDER) {
      this.modelInputResolution.width = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
      this.modelInputResolution.height = MOVENET_SINGLEPOSE_THUNDER_RESOLUTION;
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
  static async load(modelConfig: MoveNetModelConfig = MOVENET_CONFIG):
      Promise<PoseDetector> {
    const config = validateModelConfig(modelConfig);
    let model: tfc.GraphModel;
    if (config.modelUrl) {
      model = await tfc.loadGraphModel(config.modelUrl);
    } else {
      let modelUrl;
      if (config.modelType === SINGLEPOSE_LIGHTNING) {
        modelUrl = MOVENET_SINGLEPOSE_LIGHTNING_URL;
      } else if (config.modelType === SINGLEPOSE_THUNDER) {
        modelUrl = MOVENET_SINGLEPOSE_THUNDER_URL;
      }
      model = await tfc.loadGraphModel(modelUrl, {fromTFHub: true});
    }
    return new MoveNetDetector(model, config);
  }

  /**
   * Runs inference on an image using a model that is assumed to be a person
   * keypoint model that outputs 17 keypoints.
   * @param inputImage 4D tensor containing the input image. Should be of size
   *     [1, modelHeight, modelWidth, 3].
   * @param executeSync Whether to execute the model synchronously.
   * @return An InferenceResult with keypoints and scores, or null if the
   *     inference call could not be executed (for example when the model was
   *     not initialized yet) or if it produced an unexpected tensor size.
   */
  async detectKeypoints(inputImage: tf.Tensor4D, executeSync = true):
      Promise<Keypoint[]|null> {
    if (!this.moveNetModel) {
      return null;
    }

    const numKeypoints = 17;

    let outputTensor;
    if (executeSync) {
      outputTensor = this.moveNetModel.execute(inputImage) as tf.Tensor;
    } else {
      outputTensor =
          await this.moveNetModel.executeAsync(inputImage) as tf.Tensor;
    }

    // We expect an output array of shape [1, 1, 17, 3] (batch, person,
    // keypoint, coordinate + score).
    if (!outputTensor || outputTensor.shape.length !== 4 ||
        outputTensor.shape[0] !== 1 || outputTensor.shape[1] !== 1 ||
        outputTensor.shape[2] !== numKeypoints || outputTensor.shape[3] !== 3) {
      outputTensor.dispose();
      return null;
    }

    const inferenceResult = outputTensor.dataSync();
    outputTensor.dispose();

    const keypoints: Keypoint[] = [];

    for (let i = 0; i < numKeypoints; ++i) {
      keypoints[i] = {
        y: inferenceResult[i * 3],
        x: inferenceResult[i * 3 + 1],
        score: inferenceResult[i * 3 + 2]
      };
    }

    return keypoints;
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It returns a
   * single pose.
   *
   * @param image ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement
   * The input image to feed through the network.
   *
   * @param config Optional. A configuration object with the following
   * properties:
   *  `maxPoses`: Optional. Has to be set to 1.
   *  `enableSmoothing`: Optional. Defaults to `true`. When enabled, a temporal
   *  smoothing filter will be used on the keypoint locations to reduce jitter.
   *
   * @param timestamp Optional. In microseconds, i.e. 1e-6 of a second. This is
   * useful when image is a tensor, which doesn't have timestamp info. Or to
   * override timestamp in a video.
   *
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig:
          MoveNetEstimationConfig = MOVENET_SINGLE_POSE_ESTIMATION_CONFIG,
      timestamp?: number): Promise<Pose[]> {
    estimationConfig = validateEstimationConfig(estimationConfig);

    if (image == null) {
      return [];
    }

    if (timestamp == null && isVideo(image)) {
      timestamp = image.currentTime * SECOND_TO_MICRO_SECONDS;
    }

    const imageTensor3D = toImageTensor(image);
    const imageSize = getImageSize(imageTensor3D);
    const imageTensor4D: tf.Tensor4D = tf.expandDims(imageTensor3D, 0);

    // Make sure we don't dispose the input image if it's already a tensor.
    if (!(image instanceof tf.Tensor)) {
      imageTensor3D.dispose();
    }

    if (!this.cropRegion) {
      // No cropRegion was available from a previous estimatePoses() call, so
      // run the model on the full image with padding.
      let boxHeight, boxWidth;
      if (imageSize.width > imageSize.height) {
        // Create a crop region that will extend below the image, effectively
        // padding the image with a black bar at the bottom. boxHeight will be
        // larger than 1.0.
        //
        // -----------
        // |         |
        // |  image  |
        // |         |
        // -----------
        // | padding |
        // -----------
        boxHeight = imageSize.width / imageSize.height;
        boxWidth = 1.0;
      } else {
        // Create a crop region that will extend to the right of the image,
        // effectively padding the image with a black bar at the right. boxWidth
        // will be larger than 1.0.
        //
        // ---------------
        // |       |     |
        // |       | pa  |
        // | image | dd  |
        // |       | ing |
        // |       |     |
        // ---------------
        boxHeight = 1.0;
        boxWidth = imageSize.height / imageSize.width;
      }
      this.cropRegion = {
        yMin: 0.0,
        xMin: 0.0,
        yMax: boxHeight,
        xMax: boxWidth,
        height: boxHeight,
        width: boxWidth
      };
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

    let keypoints = await this.detectKeypoints(croppedImage);
    croppedImage.dispose();

    // Convert keypoints from crop coordinates to image coordinates.
    for (let i = 0; i < keypoints.length; ++i) {
      keypoints[i].y =
          this.cropRegion.yMin + keypoints[i].y * this.cropRegion.height;
      keypoints[i].x =
          this.cropRegion.xMin + keypoints[i].x * this.cropRegion.width;
    }

    // Apply the sequential filter before estimating the cropping area to make
    // it more stable.
    if (timestamp && estimationConfig.enableSmoothing) {
      keypoints = this.keypointsFilter.apply(keypoints, timestamp);
    }

    // Determine next crop region based on detected keypoints and if a crop
    // region is not detected, this will trigger the model to run on the full
    // image the next time estimatePoses() is called.
    const newCropRegion =
        this.determineCropRegion(keypoints, imageSize.height, imageSize.width);

    this.cropRegion = this.filterCropRegion(newCropRegion);

    // Convert keypoint coordinates from normalized coordinates to image space.
    for (let i = 0; i < keypoints.length; ++i) {
      keypoints[i].y *= imageSize.height;
      keypoints[i].x *= imageSize.width;
    }

    const poses: Pose[] = [];
    poses[0] = {keypoints};

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
  }

  torsoVisible(keypoints: Keypoint[]): boolean {
    return (
        keypoints[COCO_KEYPOINTS_NAMED_MAP['left_hip']].score >
            MIN_CROP_KEYPOINT_SCORE &&
        keypoints[COCO_KEYPOINTS_NAMED_MAP['right_hip']].score >
            MIN_CROP_KEYPOINT_SCORE &&
        keypoints[COCO_KEYPOINTS_NAMED_MAP['left_shoulder']].score >
            MIN_CROP_KEYPOINT_SCORE &&
        keypoints[COCO_KEYPOINTS_NAMED_MAP['right_shoulder']].score >
            MIN_CROP_KEYPOINT_SCORE);
  }

  /**
   * Calculates the maximum distance from each keypoints to the center location.
   * The function returns the maximum distances from the two sets of keypoints:
   * full 17 keypoints and 4 torso keypoints. The returned information will be
   * used to determine the crop size. See determineCropRegion for more detail.
   *
   * @param targetKeypoints Maps from joint names to coordinates.
   */
  determineTorsoAndBodyRange(
      keypoints: Keypoint[], targetKeypoints: {[index: string]: number[]},
      centerY: number, centerX: number): number[] {
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
      if (keypoints[COCO_KEYPOINTS_NAMED_MAP[key]].score <
          MIN_CROP_KEYPOINT_SCORE) {
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

  /**
   * Determines the region to crop the image for the model to run inference on.
   * The algorithm uses the detected joints from the previous frame to estimate
   * the square region that encloses the full body of the target person and
   * centers at the midpoint of two hip joints. The crop size is determined by
   * the distances between each joints and the center point.
   * When the model is not confident with the four torso joint predictions, the
   * function returns a default crop which is the full image padded to square.
   */
  determineCropRegion(
      keypoints: Keypoint[], imageHeight: number,
      imageWidth: number): BoundingBox {
    const targetKeypoints: {[index: string]: number[]} = {};

    for (const key of Object.keys(COCO_KEYPOINTS_NAMED_MAP)) {
      targetKeypoints[key] = [
        keypoints[COCO_KEYPOINTS_NAMED_MAP[key]].y * imageHeight,
        keypoints[COCO_KEYPOINTS_NAMED_MAP[key]].x * imageWidth
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
              centerX, imageWidth - centerX, centerY, imageHeight - centerY));

      let cropCorner = [centerY - cropLengthHalf, centerX - cropLengthHalf];

      if (cropLengthHalf > Math.max(imageWidth, imageHeight) / 2) {
        cropLengthHalf = Math.max(imageWidth, imageHeight) / 2;
        cropCorner = [0.0, 0.0];
      }

      const cropLength = cropLengthHalf * 2;
      return {
        yMin: cropCorner[0] / imageHeight,
        xMin: cropCorner[1] / imageWidth,
        yMax: (cropCorner[0] + cropLength) / imageHeight,
        xMax: (cropCorner[1] + cropLength) / imageWidth,
        height: (cropCorner[0] + cropLength) / imageHeight -
            cropCorner[0] / imageHeight,
        width: (cropCorner[1] + cropLength) / imageWidth -
            cropCorner[1] / imageWidth
      };
    } else {
      return null;
    }
  }
}
