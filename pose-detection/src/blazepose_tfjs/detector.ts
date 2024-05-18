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
import {calculateInverseMatrix, Matrix4x4} from '../shared/calculators/calculate_inverse_matrix';
import {calculateLandmarkProjection} from '../shared/calculators/calculate_landmark_projection';
import {calculateScoreCopy} from '../shared/calculators/calculate_score_copy';
import {calculateWorldLandmarkProjection} from '../shared/calculators/calculate_world_landmark_projection';
import {MILLISECOND_TO_MICRO_SECONDS, SECOND_TO_MICRO_SECONDS} from '../shared/calculators/constants';
import {convertImageToTensor} from '../shared/calculators/convert_image_to_tensor';
import {createSsdAnchors} from '../shared/calculators/create_ssd_anchors';
import {detectorResult} from '../shared/calculators/detector_result';
import {getImageSize, getProjectiveTransformMatrix, toImageTensor} from '../shared/calculators/image_utils';
import {ImageSize, Keypoint, Mask, Padding} from '../shared/calculators/interfaces/common_interfaces';
import {Rect} from '../shared/calculators/interfaces/shape_interfaces';
import {AnchorTensor, Detection} from '../shared/calculators/interfaces/shape_interfaces';
import {isVideo} from '../shared/calculators/is_video';
import {landmarksToDetection} from '../shared/calculators/landmarks_to_detection';
import {assertMaskValue, toHTMLCanvasElementLossy, toImageDataLossy} from '../shared/calculators/mask_util';
import {nonMaxSuppression} from '../shared/calculators/non_max_suppression';
import {normalizedKeypointsToKeypoints} from '../shared/calculators/normalized_keypoints_to_keypoints';
import {refineLandmarksFromHeatmap} from '../shared/calculators/refine_landmarks_from_heatmap';
import {removeDetectionLetterbox} from '../shared/calculators/remove_detection_letterbox';
import {removeLandmarkLetterbox} from '../shared/calculators/remove_landmark_letterbox';
import {smoothSegmentation} from '../shared/calculators/segmentation_smoothing';
import {tensorsToDetections} from '../shared/calculators/tensors_to_detections';
import {tensorsToLandmarks} from '../shared/calculators/tensors_to_landmarks';
import {tensorsToSegmentation} from '../shared/calculators/tensors_to_segmentation';
import {transformNormalizedRect} from '../shared/calculators/transform_rect';
import {KeypointsSmoothingFilter} from '../shared/filters/keypoints_smoothing';
import {LowPassVisibilityFilter} from '../shared/filters/visibility_smoothing';
import {Pose, PoseDetectorInput} from '../types';

import * as constants from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {BlazePoseTfjsEstimationConfig, BlazePoseTfjsModelConfig} from './types';

type PoseLandmarksByRoiResult = {
  landmarks: Keypoint[],
  auxiliaryLandmarks: Keypoint[],
  poseScore: number,
  worldLandmarks: Keypoint[],
  segmentationMask: tf.Tensor2D,
};

class BlazePoseTfjsMask implements Mask {
  constructor(private mask: tf.Tensor3D) {}

  async toCanvasImageSource() {
    return toHTMLCanvasElementLossy(this.mask);
  }

  async toImageData() {
    return toImageDataLossy(this.mask);
  }

  async toTensor() {
    return this.mask;
  }

  getUnderlyingType() {
    return 'tensor' as const ;
  }
}

function maskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  return 'person';
}

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
  private prevFilteredSegmentationMask: tf.Tensor2D = null;
  private visibilitySmoothingFilterActual: LowPassVisibilityFilter;
  private visibilitySmoothingFilterAuxiliary: LowPassVisibilityFilter;
  private landmarksSmoothingFilterActual: KeypointsSmoothingFilter;
  private landmarksSmoothingFilterAuxiliary: KeypointsSmoothingFilter;
  private worldLandmarksSmoothingFilterActual: KeypointsSmoothingFilter;

  constructor(
      private readonly detectorModel: tfconv.GraphModel,
      private readonly landmarkModel: tfconv.GraphModel,
      private readonly enableSmoothing: boolean,
      private enableSegmentation: boolean, private smoothSegmentation: boolean,
      private readonly modelType: BlazePoseModelType) {
    this.anchors =
        createSsdAnchors(constants.BLAZEPOSE_DETECTOR_ANCHOR_CONFIGURATION);
    const anchorW = tf.tensor1d(this.anchors.map(a => a.width));
    const anchorH = tf.tensor1d(this.anchors.map(a => a.height));
    const anchorX = tf.tensor1d(this.anchors.map(a => a.xCenter));
    const anchorY = tf.tensor1d(this.anchors.map(a => a.yCenter));
    this.anchorTensor = {x: anchorX, y: anchorY, w: anchorW, h: anchorH};
    this.prevFilteredSegmentationMask =
        this.enableSegmentation ? tf.tensor2d([], [0, 0]) : null;
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
    const poseLandmarksByRoiResult =
        await this.poseLandmarksByRoi(poseRect, image3d);

    image3d.dispose();

    if (poseLandmarksByRoiResult == null) {
      this.reset();
      return [];
    }

    const {
      landmarks: unfilteredPoseLandmarks,
      auxiliaryLandmarks: unfilteredAuxiliaryLandmarks,
      poseScore,
      worldLandmarks: unfilteredWorldLandmarks,
      segmentationMask: unfilteredSegmentationMask,
    } = poseLandmarksByRoiResult;

    // Smoothes landmarks to reduce jitter.
    const {
      actualLandmarksFiltered: poseLandmarks,
      auxiliaryLandmarksFiltered: auxiliaryLandmarks,
      actualWorldLandmarksFiltered: poseWorldLandmarks
    } =
        this.poseLandmarkFiltering(
            unfilteredPoseLandmarks, unfilteredAuxiliaryLandmarks,
            unfilteredWorldLandmarks, imageSize);

    // Calculates region of interest based on the auxiliary landmarks, to be
    // used in the subsequent image.
    const poseRectFromLandmarks =
        this.poseLandmarksToRoi(auxiliaryLandmarks, imageSize);

    // Cache roi for next image.
    this.regionOfInterest = poseRectFromLandmarks;

    // Smoothes segmentation to reduce jitter
    const filteredSegmentationMask =
        this.smoothSegmentation && unfilteredSegmentationMask != null ?
        this.poseSegmentationFiltering(unfilteredSegmentationMask) :
        unfilteredSegmentationMask;

    // Scale back keypoints.
    const keypoints = poseLandmarks != null ?
        normalizedKeypointsToKeypoints(poseLandmarks, imageSize) :
        null;

    // Add keypoint name.
    if (keypoints != null) {
      keypoints.forEach((keypoint, i) => {
        keypoint.name = BLAZEPOSE_KEYPOINTS[i];
      });
    }

    const keypoints3D = poseWorldLandmarks;

    // Add keypoint name.
    if (keypoints3D != null) {
      keypoints3D.forEach((keypoint3D, i) => {
        keypoint3D.name = BLAZEPOSE_KEYPOINTS[i];
      });
    }

    const pose: Pose = {score: poseScore, keypoints, keypoints3D};

    if (filteredSegmentationMask !== null) {
      // Grayscale to RGBA
      const rgbaMask = tf.tidy(() => {
        const mask3D =
            // tslint:disable-next-line: no-unnecessary-type-assertion
            tf.expandDims(filteredSegmentationMask, 2) as tf.Tensor3D;
        // Pads a pixel [r] to [r, 0].
        const rgMask = tf.pad(mask3D, [[0, 0], [0, 0], [0, 1]]);
        // Pads a pixel [r, 0] to [r, 0, 0, r].
        return tf.mirrorPad(rgMask, [[0, 0], [0, 0], [0, 2]], 'symmetric');
      });

      if (!this.smoothSegmentation) {
        tf.dispose(filteredSegmentationMask);
      }

      const segmentation = {
        maskValueToLabel,
        mask: new BlazePoseTfjsMask(rgbaMask)
      };

      pose.segmentation = segmentation;
    }

    return [pose];
  }

  poseSegmentationFiltering(segmentationMask: tf.Tensor2D) {
    const prevMask = this.prevFilteredSegmentationMask;
    if (prevMask.size === 0) {
      this.prevFilteredSegmentationMask = segmentationMask;
    } else {
      this.prevFilteredSegmentationMask = smoothSegmentation(
          prevMask, segmentationMask,
          constants.BLAZEPOSE_SEGMENTATION_SMOOTHING_CONFIG);
      tf.dispose(segmentationMask);
    }
    tf.dispose(prevMask);
    return this.prevFilteredSegmentationMask;
  }

  dispose() {
    this.detectorModel.dispose();
    this.landmarkModel.dispose();
    tf.dispose([
      this.anchorTensor.x, this.anchorTensor.y, this.anchorTensor.w,
      this.anchorTensor.h, this.prevFilteredSegmentationMask
    ]);
  }

  reset() {
    this.regionOfInterest = null;
    if (this.enableSegmentation) {
      tf.dispose(this.prevFilteredSegmentationMask);
      this.prevFilteredSegmentationMask = tf.tensor2d([], [0, 0]);
    }
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
    const {imageTensor: imageValueShifted, padding} = convertImageToTensor(
        image, constants.BLAZEPOSE_DETECTOR_IMAGE_TO_TENSOR_CONFIG);

    const detectionResult =
        this.detectorModel.predict(imageValueShifted) as tf.Tensor3D;
    // PoseDetectionCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 13 (data for each anchor)]
    const {boxes, logits} = detectorResult(detectionResult);

    // PoseDetectionCpu: TensorsToDetectionsCalculator
    const detections: Detection[] = await tensorsToDetections(
        [logits, boxes], this.anchorTensor,
        constants.BLAZEPOSE_TENSORS_TO_DETECTION_CONFIGURATION);

    if (detections.length === 0) {
      tf.dispose([imageValueShifted, detectionResult, logits, boxes]);
      return detections;
    }

    // PoseDetectionCpu: NonMaxSuppressionCalculator
    const selectedDetections = await nonMaxSuppression(
        detections, this.maxPoses,
        constants.BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
            .minSuppressionThreshold,
        constants.BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION
            .overlapType);

    // PoseDetectionCpu: DetectionLetterboxRemovalCalculator
    const newDetections = removeDetectionLetterbox(selectedDetections, padding);

    tf.dispose([imageValueShifted, detectionResult, logits, boxes]);

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

    // Expands pose rect with margin used during training.
    // PoseDetectionToRoi: RectTransformationCalculation.
    const roi = transformNormalizedRect(
        rawRoi, imageSize,
        constants.BLAZEPOSE_DETECTOR_RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Predict pose landmarks  and optionally segmentation within an ROI
  // subgraph: PoseLandmarksByRoiCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt
  // When poseRect is not null, image should not be null either.
  private async poseLandmarksByRoi(roi: Rect, image?: tf.Tensor3D):
      Promise<PoseLandmarksByRoiResult> {
    const imageSize = getImageSize(image);
    // Transforms the input image into a 256x256 tensor while keeping the aspect
    // ratio, resulting in potential letterboxing in the transformed image.
    const {
      imageTensor: imageValueShifted,
      padding: letterboxPadding,
      transformationMatrix
    } =
        convertImageToTensor(
            image, constants.BLAZEPOSE_LANDMARK_IMAGE_TO_TENSOR_CONFIG, roi);

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
    // activation_segmentation: This tensor (shape: [256, 256]) represents the
    // mask of the input image.
    // activation_heatmap: This tensor (shape: [1, 64, 64, 39]) represents
    //  heatmap for the 39 landmarks.
    // world_3d: This tensor (shape: [1, 117]) represents 39 3DWorld keypoints.
    const outputs =
        ['ld_3d', 'output_poseflag', 'activation_heatmap', 'world_3d'];
    if (this.enableSegmentation) {
      outputs.push('activation_segmentation');
    }

    const outputTensor =
        this.landmarkModel.execute(imageValueShifted, outputs) as tf.Tensor[];

    // Decodes the tensors into the corresponding landmark and segmentation mask
    // representation.
    // PoseLandmarksByRoiCPU: TensorsToPoseLandmarksAndSegmentation
    const tensorsToPoseLandmarksAndSegmentationResult =
        await this.tensorsToPoseLandmarksAndSegmentation(outputTensor);

    if (tensorsToPoseLandmarksAndSegmentationResult == null) {
      tf.dispose(outputTensor);
      tf.dispose(imageValueShifted);
      return null;
    }

    const {
      landmarks: roiLandmarks,
      auxiliaryLandmarks: roiAuxiliaryLandmarks,
      poseScore,
      worldLandmarks: roiWorldLandmarks,
      segmentationMask: roiSegmentationMask
    } = tensorsToPoseLandmarksAndSegmentationResult;

    const poseLandmarksAndSegmentationInverseProjectionResults =
        await this.poseLandmarksAndSegmentationInverseProjection(
            imageSize, roi, letterboxPadding, transformationMatrix,
            roiLandmarks, roiAuxiliaryLandmarks, roiWorldLandmarks,
            roiSegmentationMask);

    tf.dispose(outputTensor);
    tf.dispose(imageValueShifted);

    return {poseScore, ...poseLandmarksAndSegmentationInverseProjectionResults};
  }
  async poseLandmarksAndSegmentationInverseProjection(
      imageSize: ImageSize, roi: Rect, letterboxPadding: Padding,
      transformationMatrix: Matrix4x4, roiLandmarks: Keypoint[],
      roiAuxiliaryLandmarks: Keypoint[], roiWorldLandmarks: Keypoint[],
      roiSegmentationMask: tf.Tensor2D) {
    // -------------------------------------------------------------------------
    // ------------------------------ Landmarks --------------------------------
    // -------------------------------------------------------------------------

    // Adjusts landmarks (already normalized to [0.0, 1.0]) on the letterboxed
    // pose image to the corresponding coordinates with the letterbox removed.
    // PoseLandmarksAndSegmentationInverseProjection:
    // LandmarkLetterboxRemovalCalculator.
    const adjustedLandmarks =
        removeLandmarkLetterbox(roiLandmarks, letterboxPadding);

    // PoseLandmarksAndSegmentationInverseProjection:
    // LandmarkLetterboxRemovalCalculator.
    const adjustedAuxiliaryLandmarks =
        removeLandmarkLetterbox(roiAuxiliaryLandmarks, letterboxPadding);

    // PoseLandmarksAndSegmentationInverseProjection:
    // LandmarkProjectionCalculator.
    const landmarks = calculateLandmarkProjection(adjustedLandmarks, roi);

    const auxiliaryLandmarks =
        calculateLandmarkProjection(adjustedAuxiliaryLandmarks, roi);

    // -------------------------------------------------------------------------
    // --------------------------- World Landmarks -----------------------------
    // -------------------------------------------------------------------------

    // Projects the world landmarks from the letterboxed ROI to the full image.
    // PoseLandmarksAndSegmentationInverseProjection:
    // WorldLandmarkProjectionCalculator.
    const worldLandmarks =
        calculateWorldLandmarkProjection(roiWorldLandmarks, roi);

    // -------------------------------------------------------------------------
    // -------------------------- Segmentation Mask ----------------------------
    // -------------------------------------------------------------------------
    let segmentationMask: tf.Tensor2D|null = null;

    if (this.enableSegmentation) {
      segmentationMask = tf.tidy(() => {
        const [inputHeight, inputWidth] = roiSegmentationMask.shape;
        // Calculates the inverse transformation matrix.
        // PoseLandmarksAndSegmentationInverseProjection:
        // InverseMatrixCalculator.
        const inverseTransformationMatrix =
            calculateInverseMatrix(transformationMatrix);

        const projectiveTransform = tf.tensor2d(
            getProjectiveTransformMatrix(
                inverseTransformationMatrix,
                {width: inputWidth, height: inputHeight}, imageSize),
            [1, 8]);

        // Projects the segmentation mask from the letterboxed ROI back to the
        // full image.
        // PoseLandmarksAndSegmentationInverseProjection: WarpAffineCalculator.
        const shape4D =
            [1, inputHeight, inputWidth, 1] as [number, number, number, number];
        return tf.squeeze(
            tf.image.transform(
                tf.reshape(roiSegmentationMask, shape4D), projectiveTransform,
                'bilinear', 'constant', 0, [imageSize.height, imageSize.width]),
            [0, 3]);
      });

      tf.dispose(roiSegmentationMask);
    }

    return {landmarks, auxiliaryLandmarks, worldLandmarks, segmentationMask};
  }

  private async tensorsToPoseLandmarksAndSegmentation(tensors: tf.Tensor[]) {
    // TensorsToPoseLandmarksAndSegmentation: SplitTensorVectorCalculator.
    const landmarkTensor = tensors[0] as tf.Tensor2D,
          poseFlagTensor = tensors[1] as tf.Tensor2D,
          heatmapTensor = tensors[2] as tf.Tensor4D,
          worldLandmarkTensor = tensors[3] as tf.Tensor2D,
          segmentationTensor =
              (this.enableSegmentation ? tensors[4] : null) as tf.Tensor4D;

    // Converts the pose-flag tensor into a float that represents the
    // confidence score of pose presence.
    const poseScore = (await poseFlagTensor.data())[0];

    // Applies a threshold to the confidence score to determine whether a pose
    // is present.
    if (poseScore < constants.BLAZEPOSE_POSE_PRESENCE_SCORE) {
      return null;
    }

    // -------------------------------------------------------------------------
    // ---------------------------- Pose landmarks -----------------------------
    // -------------------------------------------------------------------------

    // Decodes the landmark tensors into a list of landmarks, where the
    // landmark coordinates are normalized by the size of the input image to
    // the model.
    // TensorsToPoseLandmarksAndSegmentation: TensorsToLandmarksCalculator.
    const rawLandmarks = await tensorsToLandmarks(
        landmarkTensor, constants.BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG);

    // Refine landmarks with heatmap tensor.
    // TensorsToPoseLandmarksAndSegmentation:
    // RefineLandmarksFromHeatmapCalculator
    const allLandmarks = await refineLandmarksFromHeatmap(
        rawLandmarks, heatmapTensor,
        constants.BLAZEPOSE_REFINE_LANDMARKS_FROM_HEATMAP_CONFIG);

    // Splits the landmarks into two sets: the actual pose landmarks and the
    // auxiliary landmarks.
    // TensorsToPoseLandmarksAndSegmentation:
    // SplitNormalizedLandmarkListCalculator
    const landmarks = allLandmarks.slice(0, constants.BLAZEPOSE_NUM_KEYPOINTS);
    const auxiliaryLandmarks = allLandmarks.slice(
        constants.BLAZEPOSE_NUM_KEYPOINTS,
        constants.BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS);

    // -------------------------------------------------------------------------
    // ------------------------- Pose world landmarks --------------------------
    // -------------------------------------------------------------------------

    // Decodes the world landmark tensors into a list of landmarks.
    // TensorsToPoseLandmarksAndSegmentation: TensorsToLandmarksCalculator.
    const allWorldLandmarks = await tensorsToLandmarks(
        worldLandmarkTensor,
        constants.BLAZEPOSE_TENSORS_TO_WORLD_LANDMARKS_CONFIG);

    // Take only actual world landmarks.
    const worldLandmarksWithoutVisibility =
        allWorldLandmarks.slice(0, constants.BLAZEPOSE_NUM_KEYPOINTS);

    const worldLandmarks =
        calculateScoreCopy(landmarks, worldLandmarksWithoutVisibility, true);

    // -------------------------------------------------------------------------
    // -------------------------- Segmentation Mask ----------------------------
    // -------------------------------------------------------------------------
    const segmentationMask: tf.Tensor2D|null = this.enableSegmentation ?
        tensorsToSegmentation(
            segmentationTensor,
            constants.BLAZEPOSE_TENSORS_TO_SEGMENTATION_CONFIG) :
        null;

    return {
      landmarks,
      auxiliaryLandmarks,
      poseScore,
      worldLandmarks,
      segmentationMask
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

    // Expands pose rect with margin used during training.
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

  const detectorFromTFHub = typeof config.detectorModelUrl === 'string' &&
      (config.detectorModelUrl.indexOf('https://tfhub.dev') > -1);
  const landmarkFromTFHub = typeof config.landmarkModelUrl === 'string' &&
      (config.landmarkModelUrl.indexOf('https://tfhub.dev') > -1);

  const [detectorModel, landmarkModel] = await Promise.all([
    tfconv.loadGraphModel(
        config.detectorModelUrl, {fromTFHub: detectorFromTFHub}),
    tfconv.loadGraphModel(
        config.landmarkModelUrl, {fromTFHub: landmarkFromTFHub})
  ]);

  return new BlazePoseTfjsDetector(
      detectorModel, landmarkModel, config.enableSmoothing,
      config.enableSegmentation, config.smoothSegmentation, config.modelType);
}
