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

import {AnchorConfig, ImageToTensorConfig, RectTransformationConfig, TensorsToDetectionsConfig, TensorsToLandmarksConfig} from '../shared/calculators/interfaces/config_interfaces';
import {MediaPipeHandsTfjsEstimationConfig, MediaPipeHandsTfjsModelConfig} from './types';

export const DEFAULT_MPHANDS_DETECTOR_MODEL_URL_LITE =
    'https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/lite/1';
export const DEFAULT_MPHANDS_DETECTOR_MODEL_URL_FULL =
    'https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/full/1';
export const DEFAULT_MPHANDS_LANDMARK_MODEL_URL_LITE =
    'https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/lite/1';
export const DEFAULT_MPHANDS_LANDMARK_MODEL_URL_FULL =
    'https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/full/1';
export const MPHANDS_DETECTOR_ANCHOR_CONFIGURATION: AnchorConfig = {
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  featureMapHeight: [] as number[],
  featureMapWidth: [] as number[],
  numLayers: 4,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 192,
  inputSizeWidth: 192,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [8, 16, 16, 16],
  aspectRatios: [1.0],
  fixedAnchorSize: true
};
export const DEFAULT_MPHANDS_MODEL_CONFIG: MediaPipeHandsTfjsModelConfig = {
  runtime: 'tfjs',
  modelType: 'full',
  maxHands: 2,
  detectorModelUrl: DEFAULT_MPHANDS_DETECTOR_MODEL_URL_FULL,
  landmarkModelUrl: DEFAULT_MPHANDS_LANDMARK_MODEL_URL_FULL
};
export const DEFAULT_MPHANDS_ESTIMATION_CONFIG:
    MediaPipeHandsTfjsEstimationConfig = {
      flipHorizontal: false,
      staticImageMode: false
    };
export const MPHANDS_TENSORS_TO_DETECTION_CONFIGURATION:
    TensorsToDetectionsConfig = {
      applyExponentialOnBoxSize: false,
      flipVertically: false,
      ignoreClasses: [] as number[],
      numClasses: 1,
      numBoxes: 2016,
      numCoords: 18,
      boxCoordOffset: 0,
      keypointCoordOffset: 4,
      numKeypoints: 7,
      numValuesPerKeypoint: 2,
      sigmoidScore: true,
      scoreClippingThresh: 100.0,
      reverseOutputOrder: true,
      xScale: 192.0,
      yScale: 192.0,
      hScale: 192.0,
      wScale: 192.0,
      minScoreThresh: 0.5
    };
export const MPHANDS_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION = {
  minSuppressionThreshold: 0.3,
  overlapType: 'intersection-over-union' as const
};
export const MPHANDS_DETECTOR_RECT_TRANSFORMATION_CONFIG:
    RectTransformationConfig = {
      shiftX: 0,
      shiftY: -0.5,
      scaleX: 2.6,
      scaleY: 2.6,
      squareLong: true
    };
export const MPHANDS_LANDMARK_RECT_TRANSFORMATION_CONFIG:
    RectTransformationConfig = {
      shiftX: 0,
      shiftY: -0.1,
      scaleX: 2.0,
      scaleY: 2.0,
      squareLong: true
    };
export const MPHANDS_DETECTOR_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 192, height: 192},
  keepAspectRatio: true,
  outputTensorFloatRange: [0, 1],
  borderMode: 'zero'
};
export const MPHANDS_LANDMARK_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 224, height: 224},
  keepAspectRatio: true,
  outputTensorFloatRange: [0, 1],
  borderMode: 'zero'
};
export const MPHANDS_HAND_PRESENCE_SCORE = 0.5;
export const MPHANDS_MIN_SIMILARITY_THRESHOLD = 0.5;
export const MPHANDS_TENSORS_TO_LANDMARKS_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 21,
  inputImageWidth: 224,
  inputImageHeight: 224,
  normalizeZ: 0.4,
  visibilityActivation: 'none',
  flipHorizontally: false,
  flipVertically: false
};
export const MPHANDS_TENSORS_TO_WORLD_LANDMARKS_CONFIG:
    TensorsToLandmarksConfig = {
      numLandmarks: 21,
      inputImageWidth: 1,
      inputImageHeight: 1,
      visibilityActivation: 'none',
      flipHorizontally: false,
      flipVertically: false
    };
