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

import {refineLandmarksFromHeatmap} from './refine_landmarks_from_heatmap';

function chwToHWC(
    chw: number[], height: number, width: number, depth: number): number[] {
  let idx = 0;
  const hwc = [];
  for (let c = 0; c < depth; ++c) {
    for (let row = 0; row < height; ++row) {
      for (let col = 0; col < width; ++col) {
        const destIdx = width * depth * row + depth * col + c;
        hwc[destIdx] = chw[idx];
        idx++;
      }
    }
  }
  return hwc;
}

describe('refineLandmarksFromHeatmap ', () => {
  const z = -10000000000000000;

  it('smoke.', async () => {
    const landmarks = [{x: 0.5, y: 0.5}];
    const heatmapTensor =
        tf.sigmoid(tf.tensor4d([z, z, z, 1, z, z, z, z, z], [1, 3, 3, 1]));

    const result = await refineLandmarksFromHeatmap(
        landmarks, heatmapTensor, {kernelSize: 3, minConfidenceToRefine: 0.1});
    expect(result[0].x).toBe(0);
    expect(result[0].y).toBe(1 / 3);
  });

  it('multi-layer.', async () => {
    const landmarks = [{x: 0.5, y: 0.5}, {x: 0.5, y: 0.5}, {x: 0.5, y: 0.5}];
    const heatmapTensor = tf.sigmoid(tf.tensor4d(
        chwToHWC(
            [
              z, z, z, 1, z, z, z, z, z, z, z, z, 1, z,
              z, z, z, z, z, z, z, 1, z, z, z, z, z
            ],
            3, 3, 3),
        [1, 3, 3, 3]));
    const result = await refineLandmarksFromHeatmap(
        landmarks, heatmapTensor, {kernelSize: 3, minConfidenceToRefine: 0.1});

    for (let i = 0; i < 3; i++) {
      expect(result[i].x).toBe(0);
      expect(result[i].y).toBe(1 / 3);
    }
  });

  it('keep if not sure.', async () => {
    const landmarks = [{x: 0.5, y: 0.5}, {x: 0.5, y: 0.5}, {x: 0.5, y: 0.5}];
    const heatmapTensor = tf.sigmoid(tf.tensor4d(
        chwToHWC(
            [
              z, z, z, 0, z, z, z, z, z, z, z, z, 0, z,
              z, z, z, z, z, z, z, 0, z, z, z, z, z
            ],
            3, 3, 3),
        [1, 3, 3, 3]));
    const result = await refineLandmarksFromHeatmap(
        landmarks, heatmapTensor, {kernelSize: 3, minConfidenceToRefine: 0.6});

    for (let i = 0; i < 3; i++) {
      expect(result[i].x).toBe(0.5);
      expect(result[i].y).toBe(0.5);
    }
  });

  it('border.', async () => {
    const landmarks = [{x: 0, y: 0}, {x: 0.9, y: 0.9}];
    const heatmapTensor = tf.sigmoid(tf.tensor4d(
        chwToHWC(
            [z, z, z, 0, z, 0, z, z, z, z, z, z, 0, z, 0, z, z, 0], 3, 3, 2),
        [1, 3, 3, 2]));

    const result = await refineLandmarksFromHeatmap(
        landmarks, heatmapTensor, {kernelSize: 3, minConfidenceToRefine: 0.1});

    expect(result[0].x).toBe(0);
    expect(result[0].y).toBe(1 / 3);
    expect(result[1].x).toBe(2 / 3);
    expect(result[1].y).toBe(1 / 6 + 2 / 6);
  });
});
