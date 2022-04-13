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

import {AnchorConfig, ImageToTensorConfig, TensorsToDetectionsConfig} from '../shared/calculators/interfaces/config_interfaces';

import {MediaPipeFaceDetectorTfjsEstimationConfig, MediaPipeFaceDetectorTfjsModelConfig} from './types';

// Non-sparse full model is not currently used but is available if needed for
// future use cases.
export const DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE =
    'https://tfhub.dev/mediapipe/tfjs-model/face_detection/full/1';
export const DEFAULT_DETECTOR_MODEL_URL_SHORT =
    'https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1';
export const SHORT_RANGE_DETECTOR_ANCHOR_CONFIG: AnchorConfig = {
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  featureMapHeight: [] as number[],
  featureMapWidth: [] as number[],
  numLayers: 4,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 128,
  inputSizeWidth: 128,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [8, 16, 16, 16],
  aspectRatios: [1.0],
  fixedAnchorSize: true
};
export const FULL_RANGE_DETECTOR_ANCHOR_CONFIG: AnchorConfig = {
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 0.0,
  featureMapHeight: [] as number[],
  featureMapWidth: [] as number[],
  numLayers: 1,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 192,
  inputSizeWidth: 192,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [4],
  aspectRatios: [1.0],
  fixedAnchorSize: true
};
export const DEFAULT_FACE_DETECTOR_MODEL_CONFIG:
    MediaPipeFaceDetectorTfjsModelConfig = {
      runtime: 'tfjs',
      modelType: 'short',
      maxFaces: 1,
      detectorModelUrl: DEFAULT_DETECTOR_MODEL_URL_SHORT,
    };
export const DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG:
    MediaPipeFaceDetectorTfjsEstimationConfig = {
      flipHorizontal: false,
    };
export const SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG:
    TensorsToDetectionsConfig = {
      applyExponentialOnBoxSize: false,
      flipVertically: false,
      ignoreClasses: [] as number[],
      numClasses: 1,
      numBoxes: 896,
      numCoords: 16,
      boxCoordOffset: 0,
      keypointCoordOffset: 4,
      numKeypoints: 6,
      numValuesPerKeypoint: 2,
      sigmoidScore: true,
      scoreClippingThresh: 100.0,
      reverseOutputOrder: true,
      xScale: 128.0,
      yScale: 128.0,
      hScale: 128.0,
      wScale: 128.0,
      minScoreThresh: 0.5
    };
export const FULL_RANGE_TENSORS_TO_DETECTION_CONFIG:
    TensorsToDetectionsConfig = {
      applyExponentialOnBoxSize: false,
      flipVertically: false,
      ignoreClasses: [] as number[],
      numClasses: 1,
      numBoxes: 2304,
      numCoords: 16,
      boxCoordOffset: 0,
      keypointCoordOffset: 4,
      numKeypoints: 6,
      numValuesPerKeypoint: 2,
      sigmoidScore: true,
      scoreClippingThresh: 100.0,
      reverseOutputOrder: true,
      xScale: 192.0,
      yScale: 192.0,
      hScale: 192.0,
      wScale: 192.0,
      minScoreThresh: 0.6
    };
export const DETECTOR_NON_MAX_SUPPRESSION_CONFIG = {
  overlapType: 'intersection-over-union' as const ,
  minSuppressionThreshold: 0.3
};
export const SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 128, height: 128},
  keepAspectRatio: true,
  outputTensorFloatRange: [-1, 1],
  borderMode: 'zero'
};
export const FULL_RANGE_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 192, height: 192},
  keepAspectRatio: true,
  outputTensorFloatRange: [-1, 1],
  borderMode: 'zero'
};
