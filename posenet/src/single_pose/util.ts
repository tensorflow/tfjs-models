/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import {NUM_KEYPOINTS} from '../keypoints';
import {Vector2D} from '../types';

export function getPointsConfidence(
    heatmapScores: tf.TensorBuffer<tf.Rank.R3>,
    heatMapCoords: tf.TensorBuffer<tf.Rank.R2>): Float32Array {
  const numKeypoints = heatMapCoords.shape[0];
  const result = new Float32Array(numKeypoints);

  for (let keypoint = 0; keypoint < numKeypoints; keypoint++) {
    const y = heatMapCoords.get(keypoint, 0);
    const x = heatMapCoords.get(keypoint, 1);
    result[keypoint] = heatmapScores.get(y, x, keypoint);
  }

  return result;
}

function getOffsetPoint(
    y: number, x: number, keypoint: number,
    offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>): Vector2D {
  return {
    y: offsetsBuffer.get(y, x, keypoint),
    x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}

export function getOffsetVectors(
    heatMapCoordsBuffer: tf.TensorBuffer<tf.Rank.R2>,
    offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>): tf.Tensor2D {
  const result: number[] = [];

  for (let keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
    const heatmapY = heatMapCoordsBuffer.get(keypoint, 0).valueOf();
    const heatmapX = heatMapCoordsBuffer.get(keypoint, 1).valueOf();

    const {x, y} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsetsBuffer);

    result.push(y);
    result.push(x);
  }

  return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
}

export function getOffsetPoints(
    heatMapCoordsBuffer: tf.TensorBuffer<tf.Rank.R2>, outputStride: number,
    offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>): tf.Tensor2D {
  return tf.tidy(() => {
    const offsetVectors = getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer);

    return tf
        .add(tf
          .cast(tf
            .mul(heatMapCoordsBuffer.toTensor(), tf.scalar(outputStride,
              'int32')), 'float32'), offsetVectors);
  });
}
