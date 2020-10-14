/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {load as loadMediaPipeFaceMesh} from './mediapipe-facemesh';

import {FaceLandmarksDetector} from './types';
export {FaceLandmarksDetector, FaceLandmarksPrediction} from './types';

export enum SupportedPackages {
  mediapipeFacemesh = 'mediapipe-facemesh'
}

/**
 * Load face-landmarks-detection.
 *
 * @param pkg - The name of the package to load, e.g. 'mediapipe-facemesh'.
 * @param config - a configuration object with the following properties:
 *  - `maxContinuousChecks` How many frames to go without running the bounding
 * box detector. Only relevant if maxFaces > 1. Defaults to 5.
 *  - `detectionConfidence` Threshold for discarding a prediction. Defaults to
 * 0.9.
 *  - `maxFaces` The maximum number of faces detected in the input. Should be
 * set to the minimum number for performance. Defaults to 10.
 *  - `iouThreshold` A float representing the threshold for deciding whether
 * boxes overlap too much in non-maximum suppression. Must be between [0, 1].
 * Defaults to 0.3.
 *  - `scoreThreshold` A threshold for deciding when to remove boxes based
 * on score in non-maximum suppression. Defaults to 0.75.
 *  - `shouldLoadIrisModel` Whether to also load the iris detection model.
 * Defaults to true.
 */
export async function load(
    pkg = SupportedPackages.mediapipeFacemesh,
    config = {}): Promise<FaceLandmarksDetector> {
  if (pkg === SupportedPackages.mediapipeFacemesh) {
    return loadMediaPipeFaceMesh(config);
  } else {
    throw new Error(`${pkg} is not a valid package name.`);
  }
}
