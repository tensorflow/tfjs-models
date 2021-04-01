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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {convertImageToTensor} from '../calculators/convert_image_to_tensor';
import {getImageSize, toImageTensor} from '../calculators/image_utils';
import {ImageSize} from '../calculators/interfaces/common_interfaces';
import {Rect} from '../calculators/interfaces/shape_interfaces';
import {isVideo} from '../calculators/is_video';
import {shiftImageValue} from '../calculators/shift_image_value';

import {BasePoseDetector, PoseDetector} from '../pose_detector';
import {Keypoint, Pose, PoseDetectorInput} from '../types';
import {calculateAlignmentPointsRects} from './calculators/calculate_alignment_points_rects';
import {calculateLandmarkProjection} from './calculators/calculate_landmark_projection';
import {createSsdAnchors} from './calculators/create_ssd_anchors';
import {detectorInference} from './calculators/detector_inference';
import {AnchorTensor, Detection} from './calculators/interfaces/shape_interfaces';
import {LandmarksSmoothingFilter} from './calculators/landmarks_smoothing';
import {landmarksToDetection} from './calculators/landmarks_to_detection';
import {nonMaxSuppression} from './calculators/non_max_suppression';
import {removeDetectionLetterbox} from './calculators/remove_detection_letterbox';
import {removeLandmarkLetterbox} from './calculators/remove_landmark_letterbox';
import {tensorsToDetections} from './calculators/tensors_to_detections';
import {tensorsToLandmarks} from './calculators/tensors_to_landmarks';
import {transformNormalizedRect} from './calculators/transform_rect';
import {LowPassVisibilityFilter} from './calculators/visibility_smoothing';
import * as constants from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {BlazeposeEstimationConfig, BlazeposeModelConfig} from './types';

type PoseLandmarkByRoiResult = {
  actualLandmarks: Keypoint[],
  auxiliaryLandmarks: Keypoint[],
  poseScore: number
};

/**
 * Blazepose detector class.
 */
export class BlazeposeDetector extends BasePoseDetector {
  private maxPoses: number;
  private upperBodyOnly: boolean;
  private anchors: Rect[];
  private anchorTensor: AnchorTensor;

  // Store global states.
  private regionOfInterest: Rect = null;
  private visibilitySmoothingFilterActual: LowPassVisibilityFilter;
  private visibilitySmoothingFilterAuxiliary: LowPassVisibilityFilter;
  private landmarksSmoothingFilterActual: LandmarksSmoothingFilter;
  private landmarksSmoothingFilterAuxiliary: LandmarksSmoothingFilter;

  // Should not be called outside.
  private constructor(
      private readonly detectorModel: tfconv.GraphModel,
      private readonly landmarkModel: tfconv.GraphModel,
      config: BlazeposeModelConfig) {
    super();

    this.upperBodyOnly = config.upperBodyOnly;

    this.anchors =
        createSsdAnchors(constants.BLAZEPOSE_DETECTOR_ANCHOR_CONFIGURATION);
    const anchorW = tf.tensor1d(this.anchors.map(a => a.width));
    const anchorH = tf.tensor1d(this.anchors.map(a => a.height));
    const anchorX = tf.tensor1d(this.anchors.map(a => a.xCenter));
    const anchorY = tf.tensor1d(this.anchors.map(a => a.yCenter));
    this.anchorTensor = {x: anchorX, y: anchorY, w: anchorW, h: anchorH};
  }

  /**
   * Loads the Blazepose model. The model to be loaded is configurable using the
   * config dictionary `BlazeposeModelConfig`. Please find more details in the
   * documentation of the `BlazeposeModelConfig`.
   *
   * @param modelConfig ModelConfig dictionary that contains parameters for
   * the Blazepose loading process. Please find more details of each parameters
   * in the documentation of the `BlazeposeModelConfig` interface.
   */
  static async load(modelConfig: BlazeposeModelConfig): Promise<PoseDetector> {
    const config = validateModelConfig(modelConfig);

    const [detectorModel, landmarkModel] = await Promise.all([
      tfconv.loadGraphModel(config.detectorModelUrl),
      tfconv.loadGraphModel(config.landmarkModelUrl)
    ]);

    return new BlazeposeDetector(detectorModel, landmarkModel, config);
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * It returns a single pose or multiple poses based on the maxPose parameter
   * from the `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config
   *       maxPoses: Optional. Max number of poses to estimate.
   *       When maxPoses = 1, a single pose is detected, it is usually much more
   *       efficient than maxPoses > 1. When maxPoses > 1, multiple poses are
   *       detected.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   *       enableSmoothing: Optional. Default to true. Smooth pose landmarks
   *       coordinates and visibility scores to reduce jitter.
   *
   * @return An array of `Pose`s.
   */
  // TF.js implementation of the mediapipe pose detection pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig: BlazeposeEstimationConfig =
          constants.DEFAULT_BLAZEPOSE_ESTIMATION_CONFIG): Promise<Pose[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.regionOfInterest = null;
      return [];
    }

    this.maxPoses = config.maxPoses;

    const imageSize = getImageSize(image);
    const image3d = tf.tidy(() => tf.cast(toImageTensor(image), 'float32'));

    let poseRect = this.regionOfInterest;

    if (poseRect == null) {
      // Need to run detector again.
      const detections = await this.detectPose(image3d);

      // Gets the very first detection from PoseDetection.
      const firstDetection = detections.length > 0 ? detections[0] : null;

      // Calculates region of interest based on pose detection, so that can be
      // used to detect landmarks.
      poseRect = this.poseDetectionToRoi(firstDetection, imageSize);
    }

    // Detects pose landmarks within specified region of interest of the image.
    const poseLandmarks = await this.poseLandmarkByRoi(poseRect, image3d);

    const {actualLandmarks, auxiliaryLandmarks, poseScore} = poseLandmarks;

    // Smoothes landmarks to reduce jitter.
    const {actualLandmarksFiltered, auxiliaryLandmarksFiltered} =
        this.poseLandmarkFiltering(
            actualLandmarks, auxiliaryLandmarks, image, config.enableSmoothing);

    // Calculates region of interest based on the auxiliary landmarks, to be
    // used in the subsequent image.
    const poseRectFromLandmarks =
        this.poseLandmarksToRoi(auxiliaryLandmarksFiltered, imageSize);

    // Cache roi for next image.
    this.regionOfInterest = poseRectFromLandmarks;

    image3d.dispose();

    const pose: Pose = {score: poseScore, keypoints: actualLandmarksFiltered};

    return [pose];
  }

  dispose() {
    this.detectorModel.dispose();
    this.landmarkModel.dispose();
    tf.dispose([
      this.anchorTensor.x, this.anchorTensor.y, this.anchorTensor.w,
      this.anchorTensor.h
    ]);
  }

  // Detects poses.
  // Subgraph: PoseDetectionCpu.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
  private async detectPose(image: PoseDetectorInput): Promise<Detection[]> {
    // PoseDetectionCpu: ImageToTensorCalculator
    // Transforms the input image into a 128x128 while keeping the aspect ratio
    // resulting in potential letterboxing in the transformed image.
    const {imageTensor, padding} = convertImageToTensor(
        image, constants.BLAZEPOSE_DETECTOR_IMAGE_TO_TENSOR_CONFIG);

    const imageValueShifted = shiftImageValue(imageTensor, [-1, 1]);

    // PoseDetectionCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 13 (data for each anchor)]
    const {boxes, scores} =
        detectorInference(imageValueShifted, this.detectorModel);

    // PoseDetectionCpu: TensorsToDetectionsCalculator
    const detections: Detection[] = await tensorsToDetections(
        [scores, boxes], this.anchorTensor,
        constants.BLAZEPOSE_TENSORS_TO_DETECTION_CONFIGURATION);

    // PoseDetectionCpu: NonMaxSuppressionCalculator
    const selectedDetections = await nonMaxSuppression(
        detections, this.maxPoses,
        constants.BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
            .minSuppressionThreshold,
        constants.BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
            .minScoreThreshold);

    // PoseDetectionCpu: DetectionLetterboxRemovalCalculator
    const newDetections = removeDetectionLetterbox(selectedDetections, padding);

    tf.dispose([imageTensor, imageValueShifted, scores, boxes]);

    return newDetections;
  }

  // Calculates region of interest from a detection.
  // Subgraph: PoseDetectionToRoi.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
  // If detection is not null, imageSize should not be null either.
  private poseDetectionToRoi(detection?: Detection, imageSize?: ImageSize):
      Rect {
    if (detection == null) {
      return null;
    }

    let startKeypointIndex;
    let endKeypointIndex;

    // Converts pose detection into a rectangle based on center and scale
    // alignment points. Pose detection contains four key points: first two for
    // full-body pose and two more for upper-body pose.
    if (this.upperBodyOnly) {
      startKeypointIndex = 2;
      endKeypointIndex = 3;
    } else {
      // full body.
      startKeypointIndex = 0;
      endKeypointIndex = 1;
    }

    // PoseDetectionToRoi: AlignmentPointsRectsCalculator.
    const rawRoi = calculateAlignmentPointsRects(detection, imageSize, {
      rotationVectorEndKeypointIndex: endKeypointIndex,
      rotationVectorStartKeypointIndex: startKeypointIndex,
      rotationVectorTargetAngleDegree: 90
    });

    // Expands pose rect with marging used during training.
    // PoseDetectionToRoi: RectTransformationCalculation.
    const roi = transformNormalizedRect(
        rawRoi, imageSize,
        constants.BLAZEPOSE_DETECTOR_RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Predict upper-body or full-body pose landmarks.
  // subgraph: PoseLandmarkByRoiCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt
  // When poseRect is not null, image should not be null either.
  private async poseLandmarkByRoi(poseRect?: Rect, image?: tf.Tensor3D):
      Promise<PoseLandmarkByRoiResult> {
    if (poseRect == null) {
      // Return empty array instead of null because downstream calculators
      // may expect an array type.
      return {actualLandmarks: null, auxiliaryLandmarks: null, poseScore: 0};
    }
    // Transforms the input image into a 256x256 tensor while keeping the aspect
    // ratio, resulting in potential letterboxing in the transformed image.
    const {imageTensor, padding} = convertImageToTensor(
        image, constants.BLAZEPOSE_LANDMARK_IMAGE_TO_TENSOR_CONFIG, poseRect);

    const imageValueShifted = shiftImageValue(imageTensor, [0, 1]);

    // PoseLandmarkByRoiCPU: InferenceCalculator
    // The model returns 4 tensor with the following shape:
    // For upperBodyOnly:
    // Only Output[1] and Output[2] matters for the pipeline.
    // Output[1]: This tensor (shape: [1, 155]) represents 31 5-d keypoints.
    // The first 25 refer to the upper body. The final 6 key points refer to
    // the alignment points from the detector model and the hands.)
    // Output [2]: This tensor (shape: [1, 1]) represents the confidence
    // score.
    // For full body:
    // Only Output[3] and Output[2] matters for the pipeline.
    // Output[3]: This tensor (shape: [1, 195]) represents 39 5-d keypoints.
    // The first 33 refer to the upper body. The final 6 key points refer to
    // the alignment points from the detector model and the hands.)
    // Output [2]: This tensor (shape: [1, 1]) represents the confidence
    // score.
    const landmarkResult =
        this.landmarkModel.predict(imageValueShifted) as tf.Tensor[];

    const landmarkTensor =
        landmarkResult[this.upperBodyOnly ? 1 : 3] as tf.Tensor2D;
    const poseFlag = landmarkResult[2] as tf.Tensor2D;

    // Converts the pose-flag tensor into a float that represents the
    // confidence score of pose presence.
    const poseScore = (await poseFlag.data())[0];

    // Applies a threshold to the confidence score to determine whether a pose
    // is present.
    if (poseScore < constants.BLAZEPOSE_POSE_PRESENCE_SCORE) {
      tf.dispose(landmarkResult);
      tf.dispose([imageTensor, imageValueShifted]);

      return {actualLandmarks: null, auxiliaryLandmarks: null, poseScore};
    }

    // Decodes the landmark tensors into a list of landmarks, where the landmark
    // coordinates are normalized by the size of the input image to the model.
    // PoseLandmarkByRoiCpu: TensorsToLandmarksCalculator.
    const landmarks = await tensorsToLandmarks(
        landmarkTensor,
        this.upperBodyOnly ?
            constants.BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG_UPPERBODY :
            constants.BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG_FULLBODY);

    // Adjusts landmarks (already normalized to [0.0, 1.0]) on the letterboxed
    // pose image to the corresponding locations on the same image with the
    // letterbox removed.
    // PoseLandmarkByRoiCpu: LandmarkLetterboxRemovalCalculator.
    const adjustedLandmarks = removeLandmarkLetterbox(landmarks, padding);

    // Projects the landmarks from the cropped pose image to the corresponding
    // locations on the full image before cropping (input to the graph).
    // PoseLandmarkByRoiCpu: LandmarkProjectionCalculator.
    const landmarksProjected =
        calculateLandmarkProjection(adjustedLandmarks, poseRect);

    // Splits the landmarks into two sets: the actual pose landmarks and the
    // auxiliary landmarks.
    const actualLandmarks = this.upperBodyOnly ?
        landmarksProjected.slice(
            0, constants.BLAZEPOSE_NUM_KEYPOINTS_UPPERBODY) :
        landmarksProjected.slice(0, constants.BLAZEPOSE_NUM_KEYPOINTS_FULLBODY);
    const auxiliaryLandmarks = this.upperBodyOnly ?
        landmarksProjected.slice(
            constants.BLAZEPOSE_NUM_KEYPOINTS_UPPERBODY,
            constants.BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS_UPPERBODY) :
        landmarksProjected.slice(
            constants.BLAZEPOSE_NUM_KEYPOINTS_FULLBODY,
            constants.BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS_FULLBODY);

    tf.dispose(landmarkResult);
    tf.dispose([imageTensor, imageValueShifted]);

    return {actualLandmarks, auxiliaryLandmarks, poseScore};
  }

  // Calculate region of interest (ROI) from landmarks.
  // Subgraph: PoseLandmarkByRoiCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt
  // When landmarks is not null, imageSize should not be null either.
  private poseLandmarksToRoi(landmarks?: Keypoint[], imageSize?: ImageSize):
      Rect {
    if (landmarks == null) {
      return null;
    }
    // PoseLandmarksToRoi: LandmarksToDetectionCalculator.
    const detection = landmarksToDetection(landmarks);

    // Converts detection into a rectangle based on center and scale alignment
    // points.
    // PoseLandmarksToRoi: AlignmentPointsRectsCalculator.
    const rawRoi = calculateAlignmentPointsRects(detection, imageSize, {
      rotationVectorStartKeypointIndex: 0,
      rotationVectorEndKeypointIndex: 1,
      rotationVectorTargetAngleDegree: 90
    });

    // Expands pose rect with marging used during training.
    // PoseLandmarksToRoi: RectTransformationCalculator.
    const roi = transformNormalizedRect(
        rawRoi, imageSize,
        constants.BLAZEPOSE_DETECTOR_RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Filter landmarks temporally to reduce jitter.
  // Subgraph: PoseLandmarkFiltering
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt
  private poseLandmarkFiltering(
      actualLandmarks: Keypoint[], auxiliaryLandmarks: Keypoint[],
      image: PoseDetectorInput, enableSmoothing: boolean): {
    actualLandmarksFiltered: Keypoint[],
    auxiliaryLandmarksFiltered: Keypoint[]
  } {
    let actualLandmarksFiltered;
    let auxiliaryLandmarksFiltered;
    if (!isVideo(image) || !enableSmoothing) {
      actualLandmarksFiltered = actualLandmarks;
      auxiliaryLandmarksFiltered = auxiliaryLandmarks;
    } else {
      // Smoothes pose landmark visibilities to reduce jitter.
      if (this.visibilitySmoothingFilterActual == null) {
        this.visibilitySmoothingFilterActual = new LowPassVisibilityFilter(
            constants.BLAZEPOSE_VISIBILITY_SMOOTHING_CONFIG);
      }
      actualLandmarksFiltered =
          this.visibilitySmoothingFilterActual.apply(actualLandmarks);

      if (this.visibilitySmoothingFilterAuxiliary == null) {
        this.visibilitySmoothingFilterAuxiliary = new LowPassVisibilityFilter(
            constants.BLAZEPOSE_VISIBILITY_SMOOTHING_CONFIG);
      }
      auxiliaryLandmarksFiltered =
          this.visibilitySmoothingFilterAuxiliary.apply(auxiliaryLandmarks);

      // Smoothes pose landmark coordinates to reduce jitter.
      if (this.landmarksSmoothingFilterActual == null) {
        this.landmarksSmoothingFilterActual = new LandmarksSmoothingFilter(
            constants.BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG);
      }
      actualLandmarksFiltered = this.landmarksSmoothingFilterActual.apply(
          actualLandmarksFiltered, image);

      if (this.landmarksSmoothingFilterAuxiliary == null) {
        this.landmarksSmoothingFilterAuxiliary = new LandmarksSmoothingFilter(
            constants.BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG);
      }
      auxiliaryLandmarksFiltered = this.landmarksSmoothingFilterAuxiliary.apply(
          auxiliaryLandmarksFiltered, image);
    }

    return {actualLandmarksFiltered, auxiliaryLandmarksFiltered};
  }
}
