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

import {AnnotatedPrediction as MediaPipePrediction, MediaPipeFaceMeshEstimateFacesConfig} from './mediapipe-facemesh';

// The object returned by a FaceLandmarksPackage describing a face found in the
// input.
export type FaceLandmarksPrediction = MediaPipePrediction;

export type EstimateFacesConfig = MediaPipeFaceMeshEstimateFacesConfig;

// The interface that defines packages that detect face landmarks in an input.
export class FaceLandmarksPackage {
  // Returns an array of faces in an image.
  estimateFaces(config: EstimateFacesConfig):
      Promise<FaceLandmarksPrediction[]> {
    throw new Error('estimateFaces is not yet implemented.');
  }
}

export {MediaPipePrediction, MediaPipeFaceMeshEstimateFacesConfig};
