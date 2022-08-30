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
import * as tf from '@tensorflow/tfjs-core';

import {Keypoint} from './interfaces/common_interfaces';

import {TensorsToLandmarksConfig} from './interfaces/config_interfaces';
import {sigmoid} from './sigmoid';

function applyActivation(activation: 'none'|'sigmoid', value: number) {
  return activation === 'none' ? value : sigmoid(value);
}

/**
 * A calculator for converting Tensors from regression models into landmarks.
 * Note that if the landmarks in the tensor has more than 5 dimensions, only the
 * first 5 dimensions will be converted to [x,y,z, visibility, presence]. The
 * latter two fields may also stay unset if such attributes are not supported in
 * the model.
 * @param landmarkTensor List of Tensors of type float32. Only the first tensor
 * will be used. The size of the values must be (num_dimension x num_landmarks).
 * @param flipHorizontally Optional. Whether to flip landmarks horizontally or
 * not. Overrides corresponding field in config.
 * @param flipVertically Optional. Whether to flip landmarks vertically or not.
 * Overrides corresponding field in config.
 *
 * @param config
 *
 * @returns Normalized landmarks.
 */
export async function tensorsToLandmarks(
    landmarkTensor: tf.Tensor, config: TensorsToLandmarksConfig,
    flipHorizontally?: boolean, flipVertically?: boolean) {
  flipHorizontally = flipHorizontally || config.flipHorizontally || false;
  flipVertically = flipVertically || config.flipVertically || false;

  const numValues = landmarkTensor.size;
  const numDimensions = numValues / config.numLandmarks;
  const rawLandmarks = await landmarkTensor.data() as Float32Array;

  const outputLandmarks: Keypoint[] = [];
  for (let ld = 0; ld < config.numLandmarks; ++ld) {
    const offset = ld * numDimensions;
    const landmark: Keypoint = {x: 0, y: 0};

    if (flipHorizontally) {
      landmark.x = config.inputImageWidth - rawLandmarks[offset];
    } else {
      landmark.x = rawLandmarks[offset];
    }
    if (numDimensions > 1) {
      if (flipVertically) {
        landmark.y = config.inputImageHeight - rawLandmarks[offset + 1];
      } else {
        landmark.y = rawLandmarks[offset + 1];
      }
    }
    if (numDimensions > 2) {
      landmark.z = rawLandmarks[offset + 2];
    }
    if (numDimensions > 3) {
      landmark.score = applyActivation(
          config.visibilityActivation, rawLandmarks[offset + 3]);
    }
    // presence is in rawLandmarks[offset + 4], we don't expose it.

    outputLandmarks.push(landmark);
  }

  for (let i = 0; i < outputLandmarks.length; ++i) {
    const landmark = outputLandmarks[i];
    landmark.x = landmark.x / config.inputImageWidth;
    landmark.y = landmark.y / config.inputImageHeight;
    // Scale Z coordinate as X + allow additional uniform normalization.
    landmark.z = landmark.z / config.inputImageWidth / (config.normalizeZ || 1);
  }

  return outputLandmarks;
}
