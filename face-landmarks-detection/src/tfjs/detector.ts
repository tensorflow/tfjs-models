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

import {createDetector, MediaPipeFaceDetectorTfjs, SupportedModels} from '@tensorflow-models/face-detection';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {MEDIAPIPE_FACE_MESH_KEYPOINTS} from '../constants';
import {FaceLandmarksDetector} from '../face_landmarks_detector';
import {calculateAssociationNormRect} from '../shared/calculators/association_norm_rect';
import {calculateLandmarkProjection} from '../shared/calculators/calculate_landmark_projection';
import {convertImageToTensor} from '../shared/calculators/convert_image_to_tensor';
import {calculateDetectionsToRects} from '../shared/calculators/detection_to_rect';
import {getImageSize, toImageTensor} from '../shared/calculators/image_utils';
import {ImageSize, Keypoint} from '../shared/calculators/interfaces/common_interfaces';
import {Rect} from '../shared/calculators/interfaces/shape_interfaces';
import {Detection} from '../shared/calculators/interfaces/shape_interfaces';
import {landmarksRefinement} from '../shared/calculators/landmarks_refinement';
import {landmarksToDetection} from '../shared/calculators/landmarks_to_detection';
import {normalizedKeypointsToKeypoints} from '../shared/calculators/normalized_keypoints_to_keypoints';
import {tensorsToLandmarks} from '../shared/calculators/tensors_to_landmarks';
import {transformNormalizedRect} from '../shared/calculators/transform_rect';
import {Face, FaceLandmarksDetectorInput} from '../types';

import * as constants from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {MediaPipeFaceMeshTfjsEstimationConfig, MediaPipeFaceMeshTfjsModelConfig} from './types';

/**
 * MediaPipFaceMesh class.
 */
class MediaPipeFaceMeshTfjsLandmarksDetector implements FaceLandmarksDetector {
  // Store global states.
  private prevFaceRectsFromLandmarks: Rect[] = null;

  constructor(
      private readonly detector: MediaPipeFaceDetectorTfjs,
      private readonly landmarkModel: tfconv.GraphModel,
      private readonly maxFaces: number,
      private readonly withAttention: boolean) {}

  /**
   * Estimates faces for an image or video frame.
   *
   * It returns a single face or multiple faces based on the maxFaces
   * parameter from the `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param estimationConfig Optional. See
   *     `MediaPipeFaceMeshTfjsEstimationConfig` documentation for detail.
   *
   * @return An array of `Face`s.
   */
  // TF.js implementation of the mediapipe face landmark pipeline.
  // ref graph:
  // https://github.com/google/mediapipe/blob/master/mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt
  async estimateFaces(
      image: FaceLandmarksDetectorInput,
      estimationConfig: MediaPipeFaceMeshTfjsEstimationConfig):
      Promise<Face[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      this.reset();
      return [];
    }

    // FaceLandmarkFrontCpu: ImagePropertiesCalculator
    // Calculate size of the image.
    const imageSize = getImageSize(image);

    const image3d = tf.tidy(() => {
      let imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (config.flipHorizontal) {
        const batchAxis = 0;
        imageTensor = tf.squeeze(
            tf.image.flipLeftRight(
                // tslint:disable-next-line: no-unnecessary-type-assertion
                tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D),
            [batchAxis]);
      }
      return imageTensor;
    });

    const prevFaceRectsFromLandmarks = this.prevFaceRectsFromLandmarks;

    let faceRectsFromDetections: Rect[];
    // Drops the incoming image if enough faces have already been identified
    // from the previous image. Otherwise, passes the incoming image through to
    // trigger a new round of face detection.
    if (config.staticImageMode || prevFaceRectsFromLandmarks == null ||
        prevFaceRectsFromLandmarks.length < this.maxFaces) {
      // FaceLandmarkFrontCpu: FaceDetectionShortRangeCpu
      // Detects faces.
      const allFaceDetections = await this.detector.detectFaces(image3d, false);

      if (allFaceDetections.length === 0) {
        this.reset();
        image3d.dispose();
        return [];
      }

      // FaceLandmarkFrontCpu: ClipDetectionVectorSizeCalculator
      // Makes sure there are no more detections than the provided maxFaces.
      // This is already done by our implementation of nonMaxSuppresion in
      // FaceDetector's detectFaces.
      const faceDetections = allFaceDetections;

      // FaceLandmarkFrontCpu: FaceDetectionFrontDetectionToRoi
      // Calculates region of interest based on face detections, so that can be
      // used to detect landmarks.
      faceRectsFromDetections = faceDetections.map(
          detection =>
              this.faceDetectionFrontDetectionToRoi(detection, imageSize));
    } else {
      faceRectsFromDetections = [];
    }

    // FaceLandmarkFrontCpu: AssociationNormRectCalculator
    // Performs association between NormalizedRect vector elements from
    // previous image and rects based on face detections from the current image.
    // This calculator ensures that the output faceRects array doesn't contain
    // overlapping regions based on the specified minSimilarityThreshold.
    const faceRects = calculateAssociationNormRect(
        [faceRectsFromDetections, prevFaceRectsFromLandmarks || []],
        constants.MIN_SIMILARITY_THRESHOLD);

    // FaceLandmarkFrontCpu: FaceLandmarkCpu
    // Detects face landmarks within specified region of interest of the image.
    const faceLandmarks = await Promise.all(
        faceRects.map(faceRect => this.faceLandmark(faceRect, image3d)));

    const faces = [];
    this.prevFaceRectsFromLandmarks = [];

    for (let i = 0; i < faceLandmarks.length; ++i) {
      const landmarks = faceLandmarks[i];

      if (landmarks == null) {
        continue;
      }

      this.prevFaceRectsFromLandmarks.push(
          this.faceLandmarksToRoi(landmarks, imageSize));

      // Scale back keypoints.
      const keypoints = normalizedKeypointsToKeypoints(landmarks, imageSize);

      // Add keypoint name.
      if (keypoints != null) {
        keypoints.forEach((keypoint, i) => {
          const name = MEDIAPIPE_FACE_MESH_KEYPOINTS.get(i);
          if (name != null) {
            keypoint.name = name;
          }
        });
      }

      const detection = landmarksToDetection(keypoints);

      faces.push(
          {keypoints, box: detection.locationData.relativeBoundingBox} as Face);
    }

    image3d.dispose();

    return faces;
  }

  dispose() {
    this.detector.dispose();
    this.landmarkModel.dispose();
  }

  reset() {
    this.detector.reset();
    this.prevFaceRectsFromLandmarks = null;
  }

  // calculates face ROI from face detection.
  // Subgraph: FaceDetectionFrontDetectionToRoi.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
  private faceDetectionFrontDetectionToRoi(
      detection: Detection, imageSize?: ImageSize): Rect {
    // Converts results of face detection into a rectangle (normalized by
    // image size) that encloses the face and is rotated such that the line
    // connecting left eye and right eye is aligned with the X-axis of the
    // rectangle.
    // FaceDetectionFrontDetectionToRoi: DetectionsToRectsCalculator.
    const rawRoi = calculateDetectionsToRects(
        detection, 'boundingbox', 'normRect', imageSize, {
          rotationVectorStartKeypointIndex: 0,  // Left eye.
          rotationVectorEndKeypointIndex: 1,    // Right eye.
          rotationVectorTargetAngleDegree: 0
        });

    // Expands and shifts the rectangle that contains the face so that it's
    // likely to cover the entire face.
    // FaceDetectionFrontDetectionToRoi: RectTransformationCalculation.
    const roi = transformNormalizedRect(
        rawRoi, imageSize, constants.RECT_TRANSFORMATION_CONFIG);

    return roi;
  }

  // Predict face landmarks.
  // subgraph: FaceLandmarkCpu
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt
  private async faceLandmark(roi: Rect, image: tf.Tensor3D) {
    // FaceLandmarkCpu: ImageToTensorCalculator
    // Transforms the input image into a 192x192 tensor.
    const {imageTensor: inputTensors} = convertImageToTensor(
        image, constants.LANDMARK_IMAGE_TO_TENSOR_CONFIG, roi);

    // FaceLandmarkCpu: InferenceCalculator
    // Runs a model takes an image tensor and
    // outputs a list of tensors representing, for instance, detection
    // boxes/keypoints and scores.
    const outputs = ['output_faceflag'].concat(
        this.withAttention ?
            [
              'output_mesh_identity', 'output_lips', 'Identity_6:0',
              'Identity_1:0', 'Identity_2:0', 'Identity_5:0'
            ] :
            ['output_mesh']);
    // The model returns 2 or 7 tensors with the following shape:
    // output_faceflag: This tensor (shape: [1, 1]) represents the
    // confidence score of the presence of a face.
    // Other outputs represents 2-d or 3-d keypoints of different parts of the
    // face.
    const outputTensors =
        this.landmarkModel.execute(inputTensors, outputs) as tf.Tensor[];

    const faceFlagTensor = outputTensors[0] as tf.Tensor2D,
          landmarkTensors = outputTensors.slice(1) as tf.Tensor4D[];

    // Converts the face-flag tensor into a float that represents the
    // confidence score of face presence.
    const facePresenceScore = (await faceFlagTensor.data())[0];

    // Applies a threshold to the confidence score to determine whether a face
    // is present.
    if (facePresenceScore < constants.FACE_PRESENCE_SCORE) {
      tf.dispose(outputTensors);
      tf.dispose(inputTensors);

      return null;
    }

    // Decodes the landmark tensors into a list of landmarks, where the
    // landmark coordinates are normalized by the size of the input image to
    // the model.
    // FaceLandmarkCpu: TensorsToFaceLandmarks /
    // TensorsToFaceLandmarksWithAttention.
    const landmarks = this.withAttention ?
        await this.tensorsToFaceLandmarksWithAttention(landmarkTensors) :
        await this.tensorsToFaceLandmarks(landmarkTensors);

    // Projects the landmarks from the cropped face image to the corresponding
    // locations on the full image before cropping.
    // FaceLandmarkCpu: WorldLandmarkProjectionCalculator.
    const faceLandmarks = calculateLandmarkProjection(landmarks, roi);

    tf.dispose(outputTensors);
    tf.dispose(inputTensors);

    return faceLandmarks;
  }

  // Transform single tensor into 468 facial landmarks.
  // subgraph: TensorsToFaceLandmarks
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt
  private async tensorsToFaceLandmarks(landmarkTensors: tf.Tensor4D[]) {
    return tensorsToLandmarks(
        landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG);
  }

  // Transform model output tensors into 478 facial landmarks with refined
  // lips, eyes and irises.
  // subgraph: TensorsToFaceLandmarks
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt
  private async tensorsToFaceLandmarksWithAttention(landmarkTensors:
                                                        tf.Tensor4D[]) {
    const meshLandmarks = await tensorsToLandmarks(
        landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG);
    const lipsLandmarks = await tensorsToLandmarks(
        landmarkTensors[1], constants.TENSORS_TO_LANDMARKS_LIPS_CONFIG);
    const leftEyeLandmarks = await tensorsToLandmarks(
        landmarkTensors[3], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG);
    const rightEyeLandmarks = await tensorsToLandmarks(
        landmarkTensors[5], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG);
    const leftIrisLandmarks = await tensorsToLandmarks(
        landmarkTensors[4], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG);
    const rightIrisLandmarks = await tensorsToLandmarks(
        landmarkTensors[2], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG);

    return landmarksRefinement(
        [
          meshLandmarks, lipsLandmarks, leftEyeLandmarks, rightEyeLandmarks,
          leftIrisLandmarks, rightIrisLandmarks
        ],
        [
          constants.LANDMARKS_REFINEMENT_MESH_CONFIG,
          constants.LANDMARKS_REFINEMENT_LIPS_CONFIG,
          constants.LANDMARKS_REFINEMENT_LEFT_EYE_CONFIG,
          constants.LANDMARKS_REFINEMENT_RIGHT_EYE_CONFIG,
          constants.LANDMARKS_REFINEMENT_LEFT_IRIS_CONFIG,
          constants.LANDMARKS_REFINEMENT_RIGHT_IRIS_CONFIG
        ]);
  }

  // Calculate face region of interest (ROI) from detections.
  // subgraph: FaceLandmarkLandmarksToRoi
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt
  private faceLandmarksToRoi(landmarks: Keypoint[], imageSize: ImageSize):
      Rect {
    // Converts face landmarks to a detection that tightly encloses all
    // landmarks.
    // FaceLandmarkLandmarksToRoi: LandmarksToDetectionCalculator.
    const faceDetection = landmarksToDetection(landmarks);
    // Converts the face detection into a rectangle (normalized by image size)
    // that encloses the face and is rotated such that the line connecting
    // left side of the left eye and right side of the right eye is aligned
    // with the X-axis of the rectangle.
    // FaceLandmarkLandmarksToRoi: DetectionsToRectsCalculator
    const faceRectFromLandmarks = calculateDetectionsToRects(
        faceDetection, 'boundingbox', 'normRect', imageSize, {
          rotationVectorStartKeypointIndex: 33,  // Left side of left eye.
          rotationVectorEndKeypointIndex: 263,   // Right side of right eye.
          rotationVectorTargetAngleDegree: 0
        });

    // Expands the face rectangle so that in the next video image it's likely
    // to still contain the face even with some motion.
    // FaceLandmarkLandmarksToRoi: RectTransformationCalculator.
    // TODO: `squareLong` in the config should be set to false in MediaPipe code
    // but is not due to a bug in their processing. Once fixed on their end,
    // split RECT_TRANSFORMATION_CONFIG into separate detector and landmark
    // configs, with landmark's config's `squareLong` set to false.
    const roi = transformNormalizedRect(
        faceRectFromLandmarks, imageSize, constants.RECT_TRANSFORMATION_CONFIG);

    return roi;
  }
}

/**
 * Loads the MediaPipeFaceMesh model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceMesh loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeFaceMeshTfjsModelConfig`
 * interface.
 */
export async function load(modelConfig: MediaPipeFaceMeshTfjsModelConfig):
    Promise<FaceLandmarksDetector> {
  const config = validateModelConfig(modelConfig);

  const landmarkFromTFHub = typeof config.landmarkModelUrl === 'string' &&
      (config.landmarkModelUrl.indexOf('https://tfhub.dev') > -1);
  const landmarkModel = await tfconv.loadGraphModel(
      config.landmarkModelUrl, {fromTFHub: landmarkFromTFHub});

  const detector = await createDetector(SupportedModels.MediaPipeFaceDetector, {
    modelType: 'short',
    maxFaces: config.maxFaces,
    detectorModelUrl: config.detectorModelUrl,
    runtime: config.runtime
  });

  return new MediaPipeFaceMeshTfjsLandmarksDetector(
      detector as MediaPipeFaceDetectorTfjs, landmarkModel, config.maxFaces,
      config.refineLandmarks);
}
