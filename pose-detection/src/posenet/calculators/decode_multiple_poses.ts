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
import {Pose} from '../../types';
import {K_LOCAL_MAXIMUM_RADIUS} from '../constants';
import {PoseNetOutputStride} from '../types';
import {buildPartWithScoreQueue} from './build_part_with_score_queue';
import {decodePose, getImageCoords, getInstanceScore, toTensorBuffers3D, withinNmsRadiusOfCorrespondingPoint} from './decode_multiple_poses_util';

/**
 * Detects multiple poses and finds their parts from part scores and
 * displacement vectors. It returns up to `maxDetections` object instance
 * detections in decreasing root score order. It works as follows: We first
 * create a priority queue with local part score maxima above
 * `scoreThreshold`, considering all parts at the same time. Then we
 * iteratively pull the top  element of the queue (in decreasing score order)
 * and treat it as a root candidate for a new object instance. To avoid
 * duplicate detections, we reject the root candidate if it is within a disk
 * of `nmsRadius` pixels from the corresponding part of a previously detected
 * instance, which is a form of part-based non-maximum suppression (NMS). If
 * the root candidate passes the NMS check, we start a new object instance
 * detection, treating the corresponding part as root and finding the
 * positions of the remaining parts by following the displacement vectors
 * along the tree-structured part graph. We assign to the newly detected
 * instance a score equal to the sum of scores of its parts which have not
 * been claimed by a previous instance (i.e., those at least `nmsRadius`
 * pixels away from the corresponding part of all previously detected
 * instances), divided by the total number of parts `numParts`.
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
 * @param displacementsFwd 3-D tensor of shape
 * `[height, width, 2 * num_edges]`, where `num_edges = num_parts - 1` is the
 * number of edges (parent-child pairs) in the tree. It contains the forward
 * displacements between consecutive part from the root towards the leaves.
 *
 * @param displacementsBwd 3-D tensor of shape
 * `[height, width, 2 * num_edges]`, where `num_edges = num_parts - 1` is the
 * number of edges (parent-child pairs) in the tree. It contains the backward
 * displacements between consecutive part from the root towards the leaves.
 *
 * @param outputStride The output stride that was used when feed-forwarding
 * through the PoseNet model.  Must be 32, 16, or 8.
 *
 * @param maxPoseDetections Maximum number of returned instance detections per
 * image.
 *
 * @param scoreThreshold Only return instance detections that have root part
 * score greater or equal to this value. Defaults to 0.5.
 *
 * @param nmsRadius Non-maximum suppression part distance. It needs to be
 * strictly positive. Two parts suppress each other if they are less than
 * `nmsRadius` pixels away. Defaults to 20.
 *
 * @return An array of poses and their scores, each containing keypoints and
 * the corresponding keypoint scores.
 */
export async function decodeMultiplePoses(
    heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D,
    displacementFwd: tf.Tensor3D, displacementBwd: tf.Tensor3D,
    outputStride: PoseNetOutputStride, maxPoseDetections: number,
    scoreThreshold = 0.5, nmsRadius = 20): Promise<Pose[]> {
  // tslint:disable-next-line: max-line-length
  const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
      await toTensorBuffers3D(
          [heatmapScores, offsets, displacementFwd, displacementBwd]);

  const poses: Pose[] = [];

  const queue = buildPartWithScoreQueue(
      scoreThreshold, K_LOCAL_MAXIMUM_RADIUS, scoresBuffer);

  const squaredNmsRadius = nmsRadius * nmsRadius;

  // Generate at most maxDetections object instances per image in
  // decreasing root part score order.
  while (poses.length < maxPoseDetections && !queue.empty()) {
    // The top element in the queue is the next root candidate.
    const root = queue.dequeue();

    // Part-based non-maximum suppression: We reject a root candidate if it
    // is within a disk of `nmsRadius` pixels from the corresponding part of
    // a previously detected instance.
    const rootImageCoords =
        getImageCoords(root.part, outputStride, offsetsBuffer);
    if (withinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.part.id)) {
      continue;
    }

    // Start a new detection instance at the position of the root.
    const keypoints = decodePose(
        root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer,
        displacementsBwdBuffer);
    const score = getInstanceScore(poses, squaredNmsRadius, keypoints);

    poses.push({keypoints, score});
  }

  return poses;
}
