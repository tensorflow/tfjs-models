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

import * as tfc from '@tensorflow/tfjs-core';

import {Keypoint} from '../types';
import {Model} from './model';

/**
 * Encapsulates a TensorFlow person keypoint model.
 */
export class KeypointModel extends Model {
  constructor() {
    super();
  }

  /**
   * Runs inference on an image using a model that is assumed to be a person
   * keypoint model that outputs 17 keypoints.
   * @param inputImage 4D tensor containing the input image. Should be of size
   *     [1, modelHeight, modelWidth, 3]. The tensor will be disposed.
   * @param executeSync Whether to execute the model synchronously.
   * @return An InferenceResult with keypoints and scores, or null if the
   *     inference call could not be executed (for example when the model was
   *     not initialized yet) or if it produced an unexpected tensor size.
   */
  async detectKeypoints(inputImage: tfc.Tensor4D, executeSync = true):
      Promise<Keypoint[]|null> {
    const inferenceResult = await super.runInference(inputImage, executeSync);
    // We expect an output array of shape [1, 1, 17, 3] (batch, person,
    // keypoint, coordinate + score).
    if (!inferenceResult || inferenceResult.shape.length !== 4 ||
        inferenceResult.shape[0] !== 1 || inferenceResult.shape[1] !== 1 ||
        inferenceResult.shape[2] !== 17 || inferenceResult.shape[3] !== 3) {
      return null;
    }
    const instanceValues = (inferenceResult.values as number[][][][])[0][0];

    const keypoints: Keypoint[] = [];

    const numKeypoints = 17;
    for (let i = 0; i < numKeypoints; ++i) {
      keypoints[i] = {
        y: instanceValues[i][0],
        x: instanceValues[i][1],
        score: instanceValues[i][2]
      };
    }

    return keypoints;
  }
}
