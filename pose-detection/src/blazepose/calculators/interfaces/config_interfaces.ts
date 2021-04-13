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
  strides?: number[];  // Strides of each output feature maps.
  aspectRatios?:
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
export interface TensorsToLandmarksCalculatorConfig {
  numLandmarks: number;
  inputImageWidth: number;
  inputImageHeight: number;
  normalizeZ?: number;
}
export interface VisibilitySmoothingConfig {
  alpha: number;  // Coefficient applied to a new value, and `1 - alpha` is
                  // applied to a stored value. Should be in [0, 1] range. The
                  // smaller the value, the smoother result and the bigger lag.
}
