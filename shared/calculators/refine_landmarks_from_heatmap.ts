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

import {RefineLandmarksFromHeatmapConfig} from './interfaces/config_interfaces';

/**
 * A calculator that refines landmarks using corresponding heatmap area.
 *
 * High level algorithm
 * For each landmark, we replace original value with a value calculated from the
 * area in heatmap close to original landmark position (the area is defined by
 * config.kernelSize). To calculate new coordinate from heatmap we calculate an
 * weighted average inside the kernel. We update the landmark if heatmap is
 * confident in it's prediction i.e. max(heatmap) in kernel is at least bigger
 * than config.minConfidenceToRefine.
 * @param landmarks List of lardmarks to refine.
 * @param heatmapTensor The heatmap for the landmarks with shape
 *     [height, width, channel]. The channel dimension has to be the same as
 *     the number of landmarks.
 * @param config The config for refineLandmarksFromHeap,
 *     see `RefineLandmarksFromHeatmapConfig` for detail.
 *
 * @returns Normalized landmarks.
 */
export async function refineLandmarksFromHeatmap(
    landmarks: Keypoint[], heatmapTensor: tf.Tensor4D,
    config: RefineLandmarksFromHeatmapConfig): Promise<Keypoint[]> {
  // tslint:disable-next-line: no-unnecessary-type-assertion
  const $heatmapTensor = tf.squeeze(heatmapTensor, [0]) as tf.Tensor3D;
  const [hmHeight, hmWidth, hmChannels] = $heatmapTensor.shape;
  if (landmarks.length !== hmChannels) {
    throw new Error(
        'Expected heatmap to have same number of channels ' +
        'as the number of landmarks. But got landmarks length: ' +
        `${landmarks.length}, heatmap length: ${hmChannels}`);
  }

  const outLandmarks = [];
  const heatmapBuf = await $heatmapTensor.buffer();

  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    const outLandmark = {...landmark};
    outLandmarks.push(outLandmark);

    const centerCol = Math.trunc(outLandmark.x * hmWidth);
    const centerRow = Math.trunc(outLandmark.y * hmHeight);
    // Point is outside of the image let's keep it intact.
    if (centerCol < 0 || centerCol >= hmWidth || centerRow < 0 ||
        centerCol >= hmHeight) {
      continue;
    }

    const offset = Math.trunc((config.kernelSize - 1) / 2);
    // Calculate area to iterate over. Note that we decrease the kernel on the
    // edges of the heatmap. Equivalent to zero border.
    const beginCol = Math.max(0, centerCol - offset);
    const endCol = Math.min(hmWidth, centerCol + offset + 1);
    const beginRow = Math.max(0, centerRow - offset);
    const endRow = Math.min(hmHeight, centerRow + offset + 1);

    let sum = 0;
    let weightedCol = 0;
    let weightedRow = 0;
    let maxValue = 0;

    // Main loop. Go over kernel and calculate weighted sum of coordinates,
    // sum of weights and max weights.
    for (let row = beginRow; row < endRow; ++row) {
      for (let col = beginCol; col < endCol; ++col) {
        const confidence = heatmapBuf.get(row, col, i);
        sum += confidence;
        maxValue = Math.max(maxValue, confidence);
        weightedCol += col * confidence;
        weightedRow += row * confidence;
      }
    }
    if (maxValue >= config.minConfidenceToRefine && sum > 0) {
      outLandmark.x = weightedCol / hmWidth / sum;
      outLandmark.y = weightedRow / hmHeight / sum;
    }
  }

  $heatmapTensor.dispose();

  return outLandmarks;
}
