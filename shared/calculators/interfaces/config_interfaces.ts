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

import {InputResolution} from './common_interfaces';

export interface ImageToTensorConfig {
  outputTensorSize: InputResolution;
  keepAspectRatio?: boolean;
  outputTensorFloatRange?: [number, number];
  borderMode: 'zero'|'replicate';
}
export interface VelocityFilterConfig {
  windowSize?: number;  // Number of value changes to keep over time. Higher
                        // value adds to lag and to stability.
  velocityScale?:
      number;  // Scale to apply to the velocity calculated over the given
               // window. With higher velocity `low pass filter` weights new new
               // values higher. Lower value adds to lag and to stability.
  minAllowedObjectScale?:
      number;  // If calculated object scale is less than given value smoothing
               // will be disabled and landmarks will be returned as is.
  disableValueScaling?:
      boolean;  // Disable value scaling based on object size and use `1.0`
                // instead. Value scale is calculated as inverse value of object
                // size. Object size is calculated as maximum side of
                // rectangular bounding box of the object in XY plane.
}
export interface OneEuroFilterConfig {
  frequency?: number;  // Frequency of incoming frames defined in seconds. Used
                       // only if can't be calculated from provided events (e.g.
                       // on the very fist frame).
  minCutOff?:
      number;  // Minimum cutoff frequency. Start by tuning this parameter while
               // keeping `beta=0` to reduce jittering to the desired level. 1Hz
               // (the default value) is a a good starting point.
  beta?: number;  // Cutoff slope. After `minCutOff` is configured, start
                  // increasing `beta` value to reduce the lag introduced by the
                  // `minCutoff`. Find the desired balance between jittering and
                  // lag.
  derivateCutOff?:
      number;  // Cutoff frequency for derivate. It is set to 1Hz in the
               // original algorithm, but can be turned to further smooth the
               // speed (i.e. derivate) on the object.
  thresholdCutOff?:
      number;  // The outlier threshold offset, will lead to a generally more
               // reactive filter that will be less likely to discount outliers.
  thresholdBeta?: number;  // The outlier threshold slope, will lead to a filter
  // that will more aggressively react whenever the
  // keypoint speed increases by being less likely to
  // consider that an observation is an outlier.
  minAllowedObjectScale?:
      number;  // If calculated object scale is less than given value smoothing
  // will be disabled and keypoints will be returned as is. This
  // value helps filter adjust to the distance to camera.
  disableValueScaling?:
      boolean;  // Disable value scaling based on object size and use `1.0`
                // instead. Value scale is calculated as inverse value of object
                // size. Object size is calculated as maximum side of
                // rectangular bounding box of the object in XY plane.
}
export interface KeypointsSmoothingConfig {
  velocityFilter?: VelocityFilterConfig;
  oneEuroFilter?: OneEuroFilterConfig;
}
export interface AnchorConfig {
  numLayers:
      number;  // Number of output feature maps to generate the anchors on.
  minScale: number;  // Min scales for generating anchor boxes on feature maps.
  maxScale: number;  // Max scales for generating anchor boxes on feature maps.
  inputSizeHeight: number;  // Size of input images.
  inputSizeWidth: number;
  anchorOffsetX: number;  // The offset for the center of anchors. The values is
                          // in the scale of stride. E.g. 0.5 meaning 0.5 *
                          // currentStride in pixels.
  anchorOffsetY: number;
  featureMapWidth:
      number[];  // Sizes of output feature maps to create anchors. Either
                 // featureMap size or stride should be provided.
  featureMapHeight: number[];
  strides: number[];  // Strides of each output feature maps.
  aspectRatios:
      number[];  // List of different aspect ratio to generate anchors.
  fixedAnchorSize?:
      boolean;  // Whether use fixed width and height (e.g. both 1.0) for each
                // anchor. This option can be used when the predicted anchor
                // width and height are in pixels.
  reduceBoxesInLowestLayer?:
      boolean;  // A boolean to indicate whether the fixed 3 boxes per location
                // is used in the lowest layer.
  interpolatedScaleAspectRatio?:
      number;  // An additional anchor is added with this aspect ratio and a
               // scale interpolated between the scale for a layer and the scale
               // for the next layer (1.0 for the last layer). This anchor is
               // not included if this value is 0.
}

export interface TensorsToDetectionsConfig {
  numClasses:
      number;  // The number of output classes predicted by the detection model.
  numBoxes:
      number;  // The number of output boxes predicted by the detection model.
  numCoords: number;  // The number of output values per boxes predicted by the
                      // detection model. The values contain bounding boxes,
                      // keypoints, etc.
  keypointCoordOffset?: number;   // The offset of keypoint coordinates in the
                                  // location tensor.
  numKeypoints?: number;          // The number of predicted keypoints.
  numValuesPerKeypoint?: number;  // The dimension of each keypoint, e.g. number
                                  // of values predicted for each keypoint.
  boxCoordOffset?: number;  // The offset of box coordinates in the location
                            // tensor.
  xScale?: number;          // Parameters for decoding SSD detection model.
  yScale?: number;          // Parameters for decoding SSD detection model.
  wScale?: number;          // Parameters for decoding SSD detection model.
  hScale?: number;          // Parameters for decoding SSD detection model.
  applyExponentialOnBoxSize?: boolean;
  reverseOutputOrder?: boolean;  // Whether to reverse the order of predicted x,
                                 // y from output. If false, the order is
                                 // [y_center, x_center, h, w], if true the
                                 // order is [x_center, y_center, w, h].
  ignoreClasses?: number[];  // The ids of classes that should be ignored during
                             // decoding the score for each predicted box.
  sigmoidScore?: boolean;
  scoreClippingThresh?: number;
  flipVertically?: boolean;  // Whether the detection coordinates from the input
                             // tensors should be flipped vertically (along the
                             // y-direction).
  minScoreThresh?:
      number;  // Score threshold for perserving decoded detections.
}
export interface DetectionToRectConfig {
  rotationVectorStartKeypointIndex: number;
  rotationVectorEndKeypointIndex: number;
  rotationVectorTargetAngleDegree?: number;  // In degrees.
  rotationVectorTargetAngle?: number;        // In radians.
}
export interface RectTransformationConfig {
  scaleX?: number;  // Scaling factor along the side of a rotated rect that
                    // was aligned with the X and Y axis before rotation
                    // respectively.
  scaleY?: number;
  rotation?: number;  // Additional rotation (counter-clockwise) around the
                      // rect center either in radians or in degrees.
  rotationDegree?: number;
  shiftX?: number;  // Shift along the side of a rotated rect that was
                    // aligned with the X and Y axis before rotation
                    // respectively. The shift is relative to the length of
                    // corresponding side. For example, for a rect with size
                    // (0.4, 0.6), with shiftX = 0.5 and shiftY = -0.5 the
                    // rect is shifted along the two sides by 0.2 and -0.3
                    // respectively.
  shiftY?: number;
  squareLong?: boolean;  // Change the final transformed rect into a square
                         // that shares the same center and rotation with
                         // the rect, and with the side of the square equal
                         // to either the long or short side of the rect
                         // respectively.
  squareShort?: boolean;
}
export interface TensorsToLandmarksConfig {
  numLandmarks: number;
  inputImageWidth: number;
  inputImageHeight: number;
  visibilityActivation: 'none'|'sigmoid';
  flipHorizontally?: boolean;
  flipVertically?: boolean;
  normalizeZ?: number;
}
export interface TensorsToSegmentationConfig {
  activation: 'none'|'sigmoid'|'softmax';
}
export interface SegmentationSmoothingConfig {
  combineWithPreviousRatio: number;
}
export interface RefineLandmarksFromHeatmapConfig {
  kernelSize?: number;
  minConfidenceToRefine: number;
}
export interface VisibilitySmoothingConfig {
  alpha: number;  // Coefficient applied to a new value, and `1 - alpha` is
                  // applied to a stored value. Should be in [0, 1] range. The
                  // smaller the value, the smoother result and the bigger lag.
}
type AssignAverage = number[];
export interface LandmarksRefinementConfig {
  indexesMapping:
      number[];  // Maps indexes of the given set of landmarks to indexes of the
                 // resulting set of landmarks. Should be non empty and contain
                 // the same amount of indexes as landmarks in the corresponding
                 // input.
  zRefinement: 'none'|'copy'|AssignAverage;  // Z refinement instructions.
}
