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

import {ImageToTensorConfig, LandmarksRefinementConfig, RectTransformationConfig, TensorsToLandmarksConfig} from '../shared/calculators/interfaces/config_interfaces';

import {MediaPipeFaceMeshTfjsEstimationConfig, MediaPipeFaceMeshTfjsModelConfig} from './types';

export const DEFAULT_LANDMARK_MODEL_URL =
    'https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/face_mesh/1';
export const DEFAULT_LANDMARK_MODEL_URL_WITH_ATTENTION =
    'https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/attention_mesh/1';
export const DEFAULT_FACE_MESH_MODEL_CONFIG:
    MediaPipeFaceMeshTfjsModelConfig = {
      runtime: 'tfjs',
      maxFaces: 1,
      refineLandmarks: false,
      // Default detectorModelUrl set by face-detection package.
      landmarkModelUrl: DEFAULT_LANDMARK_MODEL_URL
    };
export const DEFAULT_FACE_MESH_ESTIMATION_CONFIG:
    MediaPipeFaceMeshTfjsEstimationConfig = {
      flipHorizontal: false,
      staticImageMode: false
    };
export const RECT_TRANSFORMATION_CONFIG: RectTransformationConfig = {
  shiftX: 0,
  shiftY: 0,
  scaleX: 1.5,
  scaleY: 1.5,
  squareLong: true
};
export const LANDMARK_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig = {
  outputTensorSize: {width: 192, height: 192},
  outputTensorFloatRange: [0, 1],
  borderMode: 'replicate'
};
export const FACE_PRESENCE_SCORE = 0.5;
export const MIN_SIMILARITY_THRESHOLD = 0.5;
export const TENSORS_TO_LANDMARKS_MESH_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 468,
  inputImageWidth: 192,
  inputImageHeight: 192,
  visibilityActivation: 'none',
  flipHorizontally: false,
  flipVertically: false
};
export const TENSORS_TO_LANDMARKS_LIPS_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 80,
  inputImageWidth: 192,
  inputImageHeight: 192,
  visibilityActivation: 'none',
  flipHorizontally: false,
  flipVertically: false
};
export const TENSORS_TO_LANDMARKS_EYE_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 71,
  inputImageWidth: 192,
  inputImageHeight: 192,
  visibilityActivation: 'none',
  flipHorizontally: false,
  flipVertically: false
};
export const TENSORS_TO_LANDMARKS_IRIS_CONFIG: TensorsToLandmarksConfig = {
  numLandmarks: 5,
  inputImageWidth: 192,
  inputImageHeight: 192,
  visibilityActivation: 'none',
  flipHorizontally: false,
  flipVertically: false
};
export const LANDMARKS_REFINEMENT_MESH_CONFIG: LandmarksRefinementConfig = {
  indexesMapping: Array.from(Array(468).keys()),
  zRefinement: 'copy'
};
export const LANDMARKS_REFINEMENT_LIPS_CONFIG: LandmarksRefinementConfig = {
  indexesMapping: [
    // Lower outer.
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    // Upper outer(excluding corners).
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    // Lower inner.
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    // Upper inner(excluding corners).
    191, 80, 81, 82, 13, 312, 311, 310, 415,
    // Lower semi - outer.
    76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
    // Upper semi - outer(excluding corners).
    184, 74, 73, 72, 11, 302, 303, 304, 408,
    // Lower semi - inner.
    62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
    // Upper semi - inner(excluding corners).
    183, 42, 41, 38, 12, 268, 271, 272, 407
  ],
  zRefinement: 'none'
};
export const LANDMARKS_REFINEMENT_LEFT_EYE_CONFIG: LandmarksRefinementConfig = {
  indexesMapping: [
    // Lower contour.
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    // upper contour (excluding corners).
    246, 161, 160, 159, 158, 157, 173,
    // Halo x2 lower contour.
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    // Halo x2 upper contour (excluding corners).
    247, 30, 29, 27, 28, 56, 190,
    // Halo x3 lower contour.
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    // Halo x3 upper contour (excluding corners).
    113, 225, 224, 223, 222, 221, 189,
    // Halo x4 upper contour (no lower because of mesh structure) or
    // eyebrow inner contour.
    35, 124, 46, 53, 52, 65,
    // Halo x5 lower contour.
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    // Halo x5 upper contour (excluding corners) or eyebrow outer contour.
    156, 70, 63, 105, 66, 107, 55, 193
  ],
  zRefinement: 'none'
};
export const LANDMARKS_REFINEMENT_RIGHT_EYE_CONFIG:
    LandmarksRefinementConfig = {
      indexesMapping: [
        // Lower contour.
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        // Upper contour (excluding corners).
        466, 388, 387, 386, 385, 384, 398,
        // Halo x2 lower contour.
        359, 255, 339, 254, 253, 252, 256, 341, 463,
        // Halo x2 upper contour (excluding corners).
        467, 260, 259, 257, 258, 286, 414,
        // Halo x3 lower contour.
        446, 261, 448, 449, 450, 451, 452, 453, 464,
        // Halo x3 upper contour (excluding corners).
        342, 445, 444, 443, 442, 441, 413,
        // Halo x4 upper contour (no lower because of mesh structure) or
        // eyebrow inner contour.
        265, 353, 276, 283, 282, 295,
        // Halo x5 lower contour.
        372, 340, 346, 347, 348, 349, 350, 357, 465,
        // Halo x5 upper contour (excluding corners) or eyebrow outer contour.
        383, 300, 293, 334, 296, 336, 285, 417
      ],
      zRefinement: 'none'
    };
export const LANDMARKS_REFINEMENT_LEFT_IRIS_CONFIG:
    LandmarksRefinementConfig = {
      indexesMapping: [
        // Center.
        468,
        // Iris right edge.
        469,
        // Iris top edge.
        470,
        // Iris left edge.
        471,
        // Iris bottom edge.
        472
      ],
      zRefinement: [
        // Lower contour.
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        // Upper contour (excluding corners).
        246, 161, 160, 159, 158, 157, 173
      ]
    };
export const LANDMARKS_REFINEMENT_RIGHT_IRIS_CONFIG:
    LandmarksRefinementConfig = {
      indexesMapping: [
        // Center.
        473,
        // Iris right edge.
        474,
        // Iris top edge.
        475,
        // Iris left edge.
        476,
        // Iris bottom edge.
        477
      ],
      zRefinement: [
        // Lower contour.
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        // Upper contour (excluding corners).
        466, 388, 387, 386, 385, 384, 398
      ]
    };
