/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {PartWithScore} from '../types';

import {MaxHeap} from './max_heap';

function scoreIsMaximumInLocalWindow(
    keypointId: number, score: number, heatmapY: number, heatmapX: number,
    localMaximumRadius: number, scores: tf.TensorBuffer<tf.Rank.R3>): boolean {
  const [height, width] = scores.shape;

  let localMaximum = true;
  const yStart = Math.max(heatmapY - localMaximumRadius, 0);
  const yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
  for (let yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
    const xStart = Math.max(heatmapX - localMaximumRadius, 0);
    const xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
    for (let xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
      if (scores.get(yCurrent, xCurrent, keypointId) > score) {
        localMaximum = false;
        break;
      }
    }
    if (!localMaximum) {
      break;
    }
  }

  return localMaximum;
}

/**
 * Builds a priority queue with part candidate positions for a specific image in
 * the batch. For this we find all local maxima in the score maps with score
 * values above a threshold. We create a single priority queue across all parts.
 */
export function buildPartWithScoreQueue(
    scoreThreshold: number, localMaximumRadius: number,
    scores: tf.TensorBuffer<tf.Rank.R3>): MaxHeap<PartWithScore> {
  const [height, width, numKeypoints] = scores.shape;

  const queue = new MaxHeap<PartWithScore>(
      height * width * numKeypoints, ({score}) => score);

  for (let heatmapY = 0; heatmapY < height; ++heatmapY) {
    for (let heatmapX = 0; heatmapX < width; ++heatmapX) {
      for (let keypointId = 0; keypointId < numKeypoints; ++keypointId) {
        const score = scores.get(heatmapY, heatmapX, keypointId);

        // Only consider parts with score greater or equal to threshold as
        // root candidates.
        if (score < scoreThreshold) {
          continue;
        }

        // Only consider keypoints whose score is maximum in a local window.
        if (scoreIsMaximumInLocalWindow(
                keypointId, score, heatmapY, heatmapX, localMaximumRadius,
                scores)) {
          queue.enqueue({score, part: {heatmapY, heatmapX, id: keypointId}});
        }
      }
    }
  }

  return queue;
}
