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
import {Keypoint, PixelInput} from './shared/calculators/interfaces/common_interfaces';
import {BoundingBox} from './shared/calculators/interfaces/shape_interfaces';

export {Keypoint};

export enum SupportedModels {
  MediaPipeFaceDetector = 'MediaPipeFaceDetector',
}

/**
 * Common config to create the face detector.
 *
 * `maxFaces`: Optional. Default to 1. The maximum number of faces that will
 * be detected by the model. The number of returned faces can be less than the
 * maximum (for example when no faces are present in the input).
 */
export interface ModelConfig {
  maxFaces?: number;
}

/**
 * Common config for the `estimateFaces` method.
 *
 * `flipHorizontal`: Optional. Default to false. In some cases, the image is
 * mirrored, e.g. video stream from camera, flipHorizontal will flip the
 * keypoints horizontally.
 */
export interface EstimationConfig {
  flipHorizontal?: boolean;
}

/**
 * Allowed input format for the `estimateFaces` method.
 */
export type FaceDetectorInput = PixelInput;

export interface Face {
  box: BoundingBox;  // A bounding box around the detected face.
  keypoints:
      Keypoint[];  // Points of interest in the detected face, such as
                   // nose, eyes etc. MediaPipeFaceDetector has 6 keypoints.
}
