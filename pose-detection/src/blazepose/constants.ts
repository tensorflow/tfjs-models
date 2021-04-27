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

export const DEFAULT_BLAZEPOSE_DETECTOR_MODEL_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/blazepose/detector/model.json';
export const DEFAULT_BLAZEPOSE_LANDMARK_MODEL_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/blazepose/landmark/fullbody/model.json';
export const BLAZEPOSE_DETECTOR_ANCHOR_CONFIGURATION = {
  reduceBoxesInLowestlayer: false,
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
export const DEFAULT_BLAZEPOSE_ESTIMATION_CONFIG = {
  maxPoses: 1,
  flipHorizontal: false,
  enableSmoothing: true
};
export const BLAZEPOSE_DETECTION_MODEL_INPUT_RESOLUTION = {
  width: 128,
  height: 128
};
export const BLAZEPOSE_TENSORS_TO_DETECTION_CONFIGURATION = {
  applyExponentialOnBoxSize: false,
  flipVertically: false,
  ignoreClasses: [] as number[],
  numClasses: 1,
  numBoxes: 896,
  numCoords: 12,
  boxCoordOffset: 0,
  keypointCoordOffset: 4,
  numKeypoints: 4,
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
export const BLAZEPOSE_DETECTOR_NON_MAX_SUPPRESSION_CONFIGURATION = {
  minScoreThreshold: -1.0,
  minSuppressionThreshold: 0.3
};
export const BLAZEPOSE_DETECTOR_RECT_TRANSFORMATION_CONFIG = {
  shiftX: 0,
  shiftY: 0,
  scaleX: 1.5,
  scaleY: 1.5,
  squareLong: true
};
export const BLAZEPOSE_DETECTOR_IMAGE_TO_TENSOR_CONFIG = {
  inputResolution: {width: 128, height: 128},
  keepAspectRatio: true
};
export const BLAZEPOSE_LANDMARK_IMAGE_TO_TENSOR_CONFIG = {
  inputResolution: {width: 256, height: 256},
  keepAspectRatio: true
};
export const BLAZEPOSE_POSE_PRESENCE_SCORE = 0.5;
export const BLAZEPOSE_TENSORS_TO_LANDMARKS_CONFIG = {
  numLandmarks: 39,
  inputImageWidth: 256,
  inputImageHeight: 256
};
export const BLAZEPOSE_NUM_KEYPOINTS = 33;
export const BLAZEPOSE_NUM_AUXILIARY_KEYPOINTS = 35;
export const BLAZEPOSE_VISIBILITY_SMOOTHING_CONFIG = {
  alpha: 0.1
};
export const BLAZEPOSE_LANDMARKS_SMOOTHING_CONFIG = {
  velocityFilter: {
    windowSize: 5,
    velocityScale: 10,
    minAllowedObjectScale: 1e-6,
    disableValueScaling: false
  }
};
