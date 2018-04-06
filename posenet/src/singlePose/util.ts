/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

export function getPointsConfidence(
    heatmapScores: tf.Tensor3D, heatMapCoords: tf.Tensor2D) {
  const numKeypoints = heatMapCoords.shape[0];
  const result = new Float32Array(numKeypoints);

  const heatMapCoordsValues = heatMapCoords.buffer().values;

  for (let keypoint = 0; keypoint < numKeypoints; keypoint++) {
    const y = heatMapCoordsValues[keypoint * 2];
    const x = heatMapCoordsValues[keypoint * 2 + 1];
    result[keypoint] = heatmapScores.get(y, x, keypoint).valueOf();
  }

  return result;
}

function getOffsetPoint(
    y: number, x: number, keypoint: number,
    offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>) {
  return {
    y: offsetsBuffer.get(y, x, keypoint),
    x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}

export function getOffsetVectors(
    heatMapCoords: tf.Tensor2D, offsets: tf.Tensor3D) {
  const result: number[] = [];
  const offsetBuffer = offsets.buffer();

  for (let keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
    const heatmapY = heatMapCoords.get(keypoint, 0).valueOf();
    const heatmapX = heatMapCoords.get(keypoint, 1).valueOf();

    const {x, y} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsetBuffer);

    result.push(y);
    result.push(x);
  }

  return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
}

export function getOffsetPoints(
    heatMapCoords: tf.Tensor2D, outputStride: number, offsets: tf.Tensor3D) {
  const offsetVectors = getOffsetVectors(heatMapCoords, offsets);

  return heatMapCoords.mul(tf.scalar(outputStride, 'int32'))
      .toFloat()
      .add(offsetVectors);
}
