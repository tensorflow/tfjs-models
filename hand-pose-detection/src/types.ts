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

export {Keypoint};

export enum SupportedModels {
  MediaPipeHands = 'MediaPipeHands',
}

/**
 * Common config to create the hands detector.
 *
 * `maxHands`: Optional. Default to 2. The maximum number of hands that will
 * be detected by the model. The number of returned hands can be less than the
 * maximum (for example when no hands are present in the input).
 */
export interface ModelConfig {
  maxHands?: number;
}

/**
 * Common config for the `estimateHands` method.
 *
 * `flipHorizontal`: Optional. Default to false. In some cases, the image is
 * mirrored, e.g. video stream from camera, flipHorizontal will flip the
 * keypoints horizontally.
 *
 * `staticImageMode`: Optional. Default to true. If set to true, hand detection
 * will run on every input image, otherwise if set to false then detection runs
 * once and then the model simply tracks those landmarks without invoking
 * another detection until it loses track of any of the hands (ideal for
 * videos).
 */
export interface EstimationConfig {
  flipHorizontal?: boolean;
  staticImageMode?: boolean;
}

/**
 * Allowed input format for the `estimateHands` method.
 */
export type HandDetectorInput = PixelInput;

export interface Hand {
  keypoints: Keypoint[];
  keypoints3D?: Keypoint[];
  handedness: 'Left'|
      'Right';    // Note that handedness is determined assuming the input image
                  // is mirrored, i.e., taken with a front-facing/selfie camera
                  // with images flipped horizontally. If it is not the case,
                  // please swap the handedness output in the application.
  score: number;  // The probability of the handedness label.
}
