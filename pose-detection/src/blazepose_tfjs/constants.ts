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

import {ImageToTensorConfig, TensorsToLandmarksConfig} from '../shared/calculators/interfaces/config_interfaces';
import {BlazePoseTfjsModelConfig} from './types';

export const DEFAULT_BLAZEPOSE_DETECTOR_MODEL_URL =
    'https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/detector/1';
export const DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_FULL =
    'https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/full/2';
export const DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_LITE =
    'https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/lite/2';
export const DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_HEAVY =
    'https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/heavy/2';
export const BLAZEPOSE_DETECTOR_ANCHOR_CONFIGURATION = {
  reduceBoxesInLowestlayer: false,
  interpolatedScaleAspectRatio: 1.0,
  featureMapHeight: [] as number[],
  featureMapWidth: [] as number[],
  numLayers: 5,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 224,
  inputSizeWidth: 224,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [8, 16, 32, 32, 32],
  aspectRatios: [1.0],
  fixedAnchorSize: true
};
export const DEFAULT_BLAZEPOSE_MODEL_CONFIG: BlazePoseTfjsModelConfig = {
  runtime: 'tfjs',
  modelType: 'full',
  enableSmoothing: true,
  enableSegmentation: false,
  smoothSegmentation: true,
  detectorModelUrl: DEFAULT_BLAZEPOSE_DETECTOR_MODEL_URL,
  landmarkModelUrl: DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL_FULL
};
export const DEFAULT_BLAZEPOSE_ESTIMATION_CONFIG = {
  maxPoses: 1,
  flipHorizontal: false
};
export const BLAZEPOSE_TENSORS_TO_DETECTION_CONFIGURATION = {
  applyExponentialOnBoxSize: false,
  flipVertically: false,
  ignoreClasses: [] as number[],
  numClasses: 1,
  numBoxes: 2254,
  numCoords: 12,
  boxCoordOffset: 0,
  keypointCoordOffset: 4,
  numKeypoints: 4,
  numValuesPerKeypoint: 2,
  sigmoidScore: true,
  scoreClippingThresh: 100.0,
  reverseOutputOrder: true,
  xScale: 224.0,
  yScale: 224.0,
  hScale: 224.0,
  wScale: 224.0,
  minScoreThresh: 0.5
};
export const BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION = {
  minSuppressionThreshold: 0.3,
  overlapType: 'intersection-over-union' as const
};
export const BLAZEPOSE_DETECTOR_RECT_TRANSFORMATION_CONFIG = {
  shiftX: 0,
  shiftY: 0,
  scaleX: 1.25,
  scaleY: 1.25,
  squareLong: true
};
export const BLAZEPOSE_DETECTOR_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 224, height: 224},
  keepAspectRatio: true,
  outputTensorFloatRange: [-1, 1],
  borderMode: 'zero'
};
export const BLAZEPOSE_LANDMARK_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 256, height: 256},
  keepAspectRatio: true,
  outputTensorFloatRange: [0, 1],
  borderMode: 'zero'
};
export const BLAZEPOSE_POSE_PRESENCE_SCORE = 0.5;
export const BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 39,
  inputImageWidth: 256,
  inputImageHeight: 256,
  visibilityActivation: 'sigmoid',
  flipHorizontally: false,
  flipVertically: false
};
export const BLAZEPOSE_TENSORS_TO_WORLD_LANDMARKS_CONFIG:
    TensorsToLandmarksConfig = {
      numLandmarks: 39,
      inputImageWidth: 1,
      inputImageHeight: 1,
      visibilityActivation: 'sigmoid',
      flipHorizontally: false,
      flipVertically: false
    };
export const BLAZEPOSE_REFINE_LANDMARKS_FROM_HEATMAP_CONFIG = {
  kernelSize: 7,
  minConfidenceToRefine: 0.5
};
export const BLAZEPOSE_NUM_KEYPOINTS = 33;
export const BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS = 35;
export const BLAZEPOSE_VISIBILITY_SMOOTHING_CONFIG = {
  alpha: 0.1
};
export const BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG_ACTUAL = {
  oneEuroFilter: {
    frequency: 30,
    minCutOff: 0.05,  // minCutOff 0.05 results into ~0.01 alpha in landmark EMA
    // filter when landmark is static.
    beta: 80,  // beta 80 in combination with minCutOff 0.05 results into ~0.94
    // alpha in landmark EMA filter when landmark is moving fast.
    derivateCutOff: 1.0,  // derivativeCutOff 1.0 results into ~0.17 alpha in
    // landmark velocity EMA filter.,
    minAllowedObjectScale: 1e-6
  }
};
// Auxiliary landmarks are smoothed heavier than main landmarks to make ROI
// crop for pose landmarks prediction very stable when object is not moving but
// responsive enough in case of sudden movements.
export const BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG_AUXILIARY = {
  oneEuroFilter: {
    frequency: 30,
    minCutOff: 0.01,  // minCutOff 0.01 results into ~0.002 alpha in landmark
    // EMA filter when landmark is static.
    beta: 10.0,  // beta 10.0 in combination with minCutOff 0.01 results into
    // ~0.68 alpha in landmark EMA filter when landmark is moving
    // fast.
    derivateCutOff: 1.0,  // derivateCutOff 1.0 results into ~0.17 alpha in
    // landmark velocity EMA filter.
    minAllowedObjectScale: 1e-6
  }
};
export const BLAZEPOSE_WORLD_LANDMARKS_SMOOTHING_CONFIG_ACTUAL = {
  oneEuroFilter: {
    frequency: 30,
    minCutOff: 0.1,  // Min cutoff 0.1 results into ~ 0.02 alpha in landmark EMA
    // filter when landmark is static.
    beta:
        40,  // Beta 40.0 in combintation with min_cutoff 0.1 results into ~0.8
    // alpha in landmark EMA filter when landmark is moving fast.
    derivateCutOff: 1.0,  // Derivative cutoff 1.0 results into ~0.17 alpha in
    // landmark velocity EMA filter.
    minAllowedObjectScale: 1e-6,
    disableValueScaling:
        true  // As world landmarks are predicted in real world 3D coordintates
    // in meters (rather than in pixels of input image) prediction
    // scale does not depend on the pose size in the image.
  }
};
export const BLAZEPOSE_TENSORS_TO_SEGMENTATION_CONFIG = {
  activation: 'none' as
      const ,  // Sigmoid is not needed since it is already part of the model.
};
export const BLAZEPOSE_SEGMENTATION_SMOOTHING_CONFIG = {
  combineWithPreviousRatio: 0.7
};
