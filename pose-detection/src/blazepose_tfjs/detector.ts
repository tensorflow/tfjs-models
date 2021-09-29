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

import {BlazePoseModelType} from '../blazepose_mediapipe/types';
import {BLAZEPOSE_KEYPOINTS} from '../constants';
import {PoseDetector} from '../pose_detector';
import {calculateAlignmentPointsRects} from '../shared/calculators/calculate_alignment_points_rects';
import {calculateLandmarkProjection} from '../shared/calculators/calculate_landmark_projection';
import {calculateScoreCopy} from '../shared/calculators/calculate_score_copy';
import {calculateWorldLandmarkProjection} from '../shared/calculators/calculate_world_landmark_projection';
import {MILLISECOND_TO_MICRO_SECONDS, SECOND_TO_MICRO_SECONDS} from '../shared/calculators/constants';
import {convertImageToTensor} from '../shared/calculators/convert_image_to_tensor';
import {createSsdAnchors} from '../shared/calculators/create_ssd_anchors';
import {detectorInference} from '../shared/calculators/detector_inference';
import {getImageSize, toImageTensor} from '../shared/calculators/image_utils';
import {ImageSize, Keypoint} from '../shared/calculators/interfaces/common_interfaces';
import {Rect} from '../shared/calculators/interfaces/shape_interfaces';
import {AnchorTensor, Detection} from '../shared/calculators/interfaces/shape_interfaces';
import {isVideo} from '../shared/calculators/is_video';
import {landmarksToDetection} from '../shared/calculators/landmarks_to_detection';
import {nonMaxSuppression} from '../shared/calculators/non_max_suppression';
import {normalizedKeypointsToKeypoints} from '../shared/calculators/normalized_keypoints_to_keypoints';
import {refineLandmarksFromHeatmap} from '../shared/calculators/refine_landmarks_from_heatmap';
import {removeDetectionLetterbox} from '../shared/calculators/remove_detection_letterbox';
import {removeLandmarkLetterbox} from '../shared/calculators/remove_landmark_letterbox';
import {shiftImageValue} from '../shared/calculators/shift_image_value';
import {tensorsToDetections} from '../shared/calculators/tensors_to_detections';
import {tensorsToLandmarks} from '../shared/calculators/tensors_to_landmarks';
import {transformNormalizedRect} from '../shared/calculators/transform_rect';
import {KeypointsSmoothingFilter} from '../shared/filters/keypoints_smoothing';
import {LowPassVisibilityFilter} from '../shared/filters/visibility_smoothing';
import {Pose, PoseDetectorInput} from '../types';

import * as constants from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {BlazePoseTfjsEstimationConfig, BlazePoseTfjsModelConfig} from './types';

type PoseLandmarksByRoiResult = {
  actualLandmarks: Keypoint[],
  auxiliaryLandmarks: Keypoint[],
  poseScore: number,
  actualWorldLandmarks: Keypoint[],
};

/**
 * BlazePose detector class.
 */
class BlazePoseTfjsDetector implements PoseDetector {
  private readonly anchors: Rect[];
  private readonly anchorTensor: AnchorTensor;

  private maxPoses: number;
  private timestamp: number;  // In microseconds.

  // Store global states.
  private regionOfInterest: Rect = null;
  private visibilitySmoothingFilterActual: LowPassVisibilityFilter;
  private visibilitySmoothingFilterAuxiliary: LowPassVisibilityFilter;
  private landmarksSmoothingFilterActual: KeypointsSmoothingFilter;
  private landmarksSmoothingFilterAuxiliary: KeypointsSmoothingFilter;
  private worldLandmarksSmoothingFilterActual: KeypointsSmoothingFilter;

  constructor(
      private readonly detectorModel: tfconv.GraphModel,
      private readonly landmarkModel: tfconv.GraphModel,
      private readonly enableSmoothing: boolean,
      private readonly modelType: BlazePoseModelType) {
    this.anchors =
        createSsdAnchors(constants.BLAZEPOSE_DETECTOR_ANCHOR_CONFIGURATION);
    const anchorW = tf.tensor1d(this.anchors.map(a => a.width));
    const anchorH = tf.tensor1d(this.anchors.map(a => a.height));
    const anchorX = tf.tensor1d(this.anchors.map(a => a.xCenter));
    const anchorY = tf.tensor1d(this.anchors.map(a => a.yCenter));
    this.anchorTensor = {x: anchorX, y: anchorY, w: anchorW, h: anchorH};
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
   * @param estimationConfig Optional. See `BlazePoseTfjsEstimationConfig`
   *       documentation for detail.
   *
   * @param timestamp Optional. In milliseconds. This is useful when image is
   *     a tensor, which doesn't have timestamp info. Or to override timestamp
   *     in a video.
   *
   * @return An array of `Pose`s.
   */
  // TF.js implementation of the mediapipe pose detection pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt
  async estimatePoses(
      image: PoseDetectorInput, estimationConfig: BlazePoseTfjsEstimationConfig,
      timestamp?: number): Promise<Pose[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    this.maxPoses = config.maxPoses;

    // User provided timestamp will override video's timestamp.
    if (timestamp != null) {
      this.timestamp = timestamp * MILLISECOND_TO_MICRO_SECONDS;
    } else {
      // For static images, timestamp should be null.
      this.timestamp =
          isVideo(image) ? image.currentTime * SECOND_TO_MICRO_SECONDS : null;
    }

    const imageSize = getImageSize(image);
    const image3d = tf.tidy(() => tf.cast(toImageTensor(image), 'float32'));

    let poseRect = this.regionOfInterest;

    if (poseRect == null) {
      // Need to run detector again.
      const detections = await this.detectPose(image3d);

      if (detections.length === 0) {
        this.reset();
        image3d.dispose();
        return [];
      }

      // Gets the very first detection from PoseDetection.
      const firstDetection = detections[0];

      // Calculates region of interest based on pose detection, so that can be
      // used to detect landmarks.
      poseRect = this.poseDetectionToRoi(firstDetection, imageSize);
    }

    // Detects pose landmarks within specified region of interest of the image.
    const poseLandmarks = await this.poseLandmarksByRoi(poseRect, image3d);

    image3d.dispose();

    if (poseLandmarks == null) {
      this.reset();
      return [];
    }

    const {
      actualLandmarks,
      auxiliaryLandmarks,
      poseScore,
      actualWorldLandmarks
    } = poseLandmarks;

    // Smoothes landmarks to reduce jitter.
    const {
      actualLandmarksFiltered,
      auxiliaryLandmarksFiltered,
      actualWorldLandmarksFiltered
    } =
        this.poseLandmarkFiltering(
            actualLandmarks, auxiliaryLandmarks, actualWorldLandmarks,
            imageSize);

    // Calculates region of interest based on the auxiliary landmarks, to be
    // used in the subsequent image.
    const poseRectFromLandmarks =
        this.poseLandmarksToRoi(auxiliaryLandmarksFiltered, imageSize);

    // Cache roi for next image.
    this.regionOfInterest = poseRectFromLandmarks;

    // Scale back keypoints.
    const keypoints = actualLandmarksFiltered != null ?
        normalizedKeypointsToKeypoints(actualLandmarksFiltered, imageSize) :
        null;

    // Add keypoint name.
    if (keypoints != null) {
      keypoints.forEach((keypoint, i) => {
        keypoint.name = BLAZEPOSE_KEYPOINTS[i];
      });
    }

    const keypoints3D = actualWorldLandmarksFiltered;

    // Add keypoint name.
    if (keypoints3D != null) {
      keypoints3D.forEach((keypoint3D, i) => {
        keypoint3D.name = BLAZEPOSE_KEYPOINTS[i];
      });
    }

    const pose: Pose = {score: poseScore, keypoints, keypoints3D};

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

  reset() {
    this.regionOfInterest = null;
    this.visibilitySmoothingFilterActual = null;
    this.visibilitySmoothingFilterAuxiliary = null;
    this.landmarksSmoothingFilterActual = null;
    this.landmarksSmoothingFilterAuxiliary = null;
  }

  // Detects poses.
  // Subgraph: PoseDetectionCpu.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
  private async detectPose(image: PoseDetectorInput): Promise<Detection[]> {
    // PoseDetectionCpu: ImageToTensorCalculator
    // Transforms the input image into a 224x224 while keeping the aspect ratio
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
  private poseDetectionToRoi(detection: Detection, imageSize?: ImageSize):
      Rect {
    let startKeypointIndex;
    let endKeypointIndex;

    // Converts pose detection into a rectangle based on center and scale
    // alignment points.
    startKeypointIndex = 0;
    endKeypointIndex = 1;

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

  // Predict pose landmarks.
  // subgraph: PoseLandmarksByRoiCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt
  // When poseRect is not null, image should not be null either.
  private async poseLandmarksByRoi(poseRect: Rect, image?: tf.Tensor3D):
      Promise<PoseLandmarksByRoiResult> {
    // Transforms the input image into a 256x256 tensor while keeping the aspect
    // ratio, resulting in potential letterboxing in the transformed image.
    const {imageTensor, padding} = convertImageToTensor(
        image, constants.BLAZEPOSE_LANDMARK_IMAGE_TO_TENSOR_CONFIG, poseRect);

    const imageValueShifted = shiftImageValue(imageTensor, [0, 1]);

    if (this.modelType !== 'lite' && this.modelType !== 'full' &&
        this.modelType !== 'heavy') {
      throw new Error(
          'Model type must be one of lite, full or heavy,' +
          `but got ${this.modelType}`);
    }

    // PoseLandmarksByRoiCPU: InferenceCalculator
    // The model returns 5 tensors with the following shape:
    // ld_3d: This tensor (shape: [1, 195]) represents 39 5-d keypoints.
    // output_poseflag: This tensor (shape: [1, 1]) represents the confidence
    //  score.
    // activation_heatmap: This tensor (shape: [1, 64, 64, 39]) represents
    //  heatmap for the 39 landmarks.
    // world_3d: This tensor (shape: [1, 117]) represents 39 3DWorld keypoints.
    const landmarkResult = this.landmarkModel.execute(imageValueShifted, [
      'ld_3d', 'output_poseflag', 'activation_heatmap', 'world_3d'
    ]) as tf.Tensor[];

    const landmarkTensor = landmarkResult[0] as tf.Tensor2D,
          poseFlagTensor = landmarkResult[1] as tf.Tensor2D,
          heatmapTensor = landmarkResult[2] as tf.Tensor4D,
          worldLandmarkTensor = landmarkResult[3] as tf.Tensor2D;

    // Converts the pose-flag tensor into a float that represents the
    // confidence score of pose presence.
    const poseScore = (await poseFlagTensor.data())[0];

    // Applies a threshold to the confidence score to determine whether a pose
    // is present.
    if (poseScore < constants.BLAZEPOSE_POSE_PRESENCE_SCORE) {
      tf.dispose(landmarkResult);
      tf.dispose([imageTensor, imageValueShifted]);

      return null;
    }

    // -------------------------------------------------------------------------
    // ---------------------------- Pose landmarks -----------------------------
    // -------------------------------------------------------------------------

    // Decodes the landmark tensors into a list of landmarks, where the
    // landmark coordinates are normalized by the size of the input image to
    // the model.
    // PoseLandmarksByRoiCpu: TensorsToLandmarksCalculator.
    const landmarks = await tensorsToLandmarks(
        landmarkTensor, constants.BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG);

    // Refine landmarks with heatmap tensor.
    const refinedLandmarks = await refineLandmarksFromHeatmap(
        landmarks, heatmapTensor,
        constants.BLAZEPOSE_REFINE_LANDMARKS_FROM_HEATMAP_CONFIG);

    // Adjusts landmarks (already normalized to [0.0, 1.0]) on the letterboxed
    // pose image to the corresponding locations on the same image with the
    // letterbox removed.
    // PoseLandmarksByRoiCpu: LandmarkLetterboxRemovalCalculator.
    const adjustedLandmarks =
        removeLandmarkLetterbox(refinedLandmarks, padding);

    // Projects the landmarks from the cropped pose image to the corresponding
    // locations on the full image before cropping (input to the graph).
    // PoseLandmarksByRoiCpu: LandmarkProjectionCalculator.
    const landmarksProjected =
        calculateLandmarkProjection(adjustedLandmarks, poseRect);

    // Splits the landmarks into two sets: the actual pose landmarks and the
    // auxiliary landmarks.
    const actualLandmarks =
        landmarksProjected.slice(0, constants.BLAZEPOSE_NUM_KEYPOINTS);
    const auxiliaryLandmarks = landmarksProjected.slice(
        constants.BLAZEPOSE_NUM_KEYPOINTS,
        constants.BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS);

    // -------------------------------------------------------------------------
    // ------------------------- Pose world landmarks --------------------------
    // -------------------------------------------------------------------------

    // Decodes the world landmark tensors into a list of landmarks.
    // PoseLandmarksByRoiCpu: TensorsToLandmarksCalculator.
    const worldLandmarks = await tensorsToLandmarks(
        worldLandmarkTensor,
        constants.BLAZEPOSE_TENSORS_TO_WORLD_LANDMARKS_CONFIG);

    const worldLandmarksWithVisibility =
        calculateScoreCopy(landmarks, worldLandmarks, true);

    // Projects the world landmarks from the cropped pose image to the
    // corresponding locations on the full image before cropping (input to the
    // graph).
    // PoseLandmarksByRoiCpu: WorldLandmarkProjectionCalculator.
    const projectedWorldLandmarks = calculateWorldLandmarkProjection(
        worldLandmarksWithVisibility, poseRect);

    // Take only actual world landmarks.
    const actualWorldLandmarks =
        projectedWorldLandmarks.slice(0, constants.BLAZEPOSE_NUM_KEYPOINTS);

    tf.dispose(landmarkResult);
    tf.dispose([imageTensor, imageValueShifted]);

    return {
      actualLandmarks,
      auxiliaryLandmarks,
      poseScore,
      actualWorldLandmarks
    };
  }

  // Calculate region of interest (ROI) from landmarks.
  // Subgraph: PoseLandmarksToRoiCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt
  // When landmarks is not null, imageSize should not be null either.
  private poseLandmarksToRoi(landmarks: Keypoint[], imageSize?: ImageSize):
      Rect {
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
      actualWorldLandmarks: Keypoint[], imageSize: ImageSize): {
    actualLandmarksFiltered: Keypoint[],
    auxiliaryLandmarksFiltered: Keypoint[],
    actualWorldLandmarksFiltered: Keypoint[],
  } {
    let actualLandmarksFiltered;
    let auxiliaryLandmarksFiltered;
    let actualWorldLandmarksFiltered;
    if (this.timestamp == null || !this.enableSmoothing) {
      actualLandmarksFiltered = actualLandmarks;
      auxiliaryLandmarksFiltered = auxiliaryLandmarks;
      actualWorldLandmarksFiltered = actualWorldLandmarks;
    } else {
      const auxDetection = landmarksToDetection(auxiliaryLandmarks);
      const objectScaleROI =
          calculateAlignmentPointsRects(auxDetection, imageSize, {
            rotationVectorEndKeypointIndex: 0,
            rotationVectorStartKeypointIndex: 1,
            rotationVectorTargetAngleDegree: 90
          });

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

      actualWorldLandmarksFiltered =
          this.visibilitySmoothingFilterActual.apply(actualWorldLandmarks);

      // Smoothes pose landmark coordinates to reduce jitter.
      if (this.landmarksSmoothingFilterActual == null) {
        this.landmarksSmoothingFilterActual = new KeypointsSmoothingFilter(
            constants.BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG_ACTUAL);
      }
      actualLandmarksFiltered = this.landmarksSmoothingFilterActual.apply(
          actualLandmarksFiltered, this.timestamp, imageSize,
          true /* normalized */, objectScaleROI);

      if (this.landmarksSmoothingFilterAuxiliary == null) {
        this.landmarksSmoothingFilterAuxiliary = new KeypointsSmoothingFilter(
            constants.BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG_AUXILIARY);
      }
      auxiliaryLandmarksFiltered = this.landmarksSmoothingFilterAuxiliary.apply(
          auxiliaryLandmarksFiltered, this.timestamp, imageSize,
          true /* normalized */, objectScaleROI);

      // Smoothes pose world landmark coordinates to reduce jitter.
      if (this.worldLandmarksSmoothingFilterActual == null) {
        this.worldLandmarksSmoothingFilterActual = new KeypointsSmoothingFilter(
            constants.BLAZEPOSE_WORLD_LANDMARKS_SMOOTHING_CONFIG_ACTUAL);
      }
      actualWorldLandmarksFiltered =
          this.worldLandmarksSmoothingFilterActual.apply(
              actualWorldLandmarks, this.timestamp);
    }

    return {
      actualLandmarksFiltered,
      auxiliaryLandmarksFiltered,
      actualWorldLandmarksFiltered
    };
  }
}

/**
 * Loads the BlazePose model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the BlazePose loading process. Please find more details of each parameters
 * in the documentation of the `BlazePoseTfjsModelConfig` interface.
 */
export async function load(modelConfig: BlazePoseTfjsModelConfig):
    Promise<PoseDetector> {
  const config = validateModelConfig(modelConfig);

  const detectorFromTFHub =
      (config.detectorModelUrl.indexOf('https://tfhub.dev') > -1);
  const landmarkFromTFHub =
      (config.landmarkModelUrl.indexOf('https://tfhub.dev') > -1);

  const [detectorModel, landmarkModel] = await Promise.all([
    tfconv.loadGraphModel(
        config.detectorModelUrl, {fromTFHub: detectorFromTFHub}),
    tfconv.loadGraphModel(
        config.landmarkModelUrl, {fromTFHub: landmarkFromTFHub})
  ]);

  return new BlazePoseTfjsDetector(
      detectorModel, landmarkModel, config.enableSmoothing, config.modelType);
}
