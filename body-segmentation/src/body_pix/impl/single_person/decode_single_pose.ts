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

import {PART_NAMES} from '../keypoints';
import {BodyPixOutputStride} from '../types';
import {Keypoint, Pose} from '../types';

import {argmax2d} from './argmax2d';
import {getOffsetPoints, getPointsConfidence} from './util';

/**
 * Detects a single pose and finds its parts from part scores and offset
 * vectors. It returns a single pose detection. It works as follows:
 * argmax2d is done on the scores to get the y and x index in the heatmap
 * with the highest score for each part, which is essentially where the
 * part is most likely to exist. This produces a tensor of size 17x2, with
 * each row being the y and x index in the heatmap for each keypoint.
 * The offset vector for each for each part is retrieved by getting the
 * y and x from the offsets corresponding to the y and x index in the
 * heatmap for that part. This produces a tensor of size 17x2, with each
 * row being the offset vector for the corresponding keypoint.
 * To get the keypoint, each part’s heatmap y and x are multiplied
 * by the output stride then added to their corresponding offset vector,
 * which is in the same scale as the original image. 
 *
 * @param heatmapScores 3-D tensor with shape `[height, width, numParts]`.
 * The value of heatmapScores[y, x, k]` is the score of placing the `k`-th
 * object part at position `(y, x)`.
 *
 * @param offsets 3-D tensor with shape `[height, width, numParts * 2]`.
 * The value of [offsets[y, x, k], offsets[y, x, k + numParts]]` is the
 * short range offset vector of the `k`-th  object part at heatmap
 * position `(y, x)`.
 *
 * @param outputStride The output stride that was used when feed-forwarding
 * through the PoseNet model.  Must be 32, 16, or 8.
 *
 * @return A promise that resolves with single pose with a confidence score,
 * which contains an array of keypoints indexed by part id, each with a score
 * and position.
 */
export async function decodeSinglePose(
    heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D,
    outputStride: BodyPixOutputStride): Promise<Pose> {
  let totalScore = 0.0;

  const heatmapValues = argmax2d(heatmapScores);

  const [scoresBuffer, offsetsBuffer, heatmapValuesBuffer] = await Promise.all(
      [heatmapScores.buffer(), offsets.buffer(), heatmapValues.buffer()]);

  const offsetPoints =
      getOffsetPoints(heatmapValuesBuffer, outputStride, offsetsBuffer);
  const offsetPointsBuffer = await offsetPoints.buffer();

  const keypointConfidence =
      Array.from(getPointsConfidence(scoresBuffer, heatmapValuesBuffer));

  const keypoints = keypointConfidence.map((score, keypointId): Keypoint => {
    totalScore += score;
    return {
      position: {
        y: offsetPointsBuffer.get(keypointId, 0),
        x: offsetPointsBuffer.get(keypointId, 1)
      },
      part: PART_NAMES[keypointId],
      score
    };
  });

  heatmapValues.dispose();
  offsetPoints.dispose();

  return {keypoints, score: totalScore / keypoints.length};
}
