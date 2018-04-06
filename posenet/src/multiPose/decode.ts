/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {Keypoint, Pose} from '../types';

import {buildPartWithScoreQueue} from './buildPartWithScoreQueue';
import {decodePose} from './decodePose';
import {getImageCoords, squaredDistance} from './util';

function withinNmsRadiusOfCorrespondingPoint(
    poses: Pose[], squaredNmsRadius: number, {x, y}: {x: number, y: number},
    keypointId: number) {
  return poses.some(({keypoints}) => {
    const correspondingKeypoint = keypoints[keypointId].point;
    return squaredDistance(
               y, x, correspondingKeypoint.y, correspondingKeypoint.x) <=
        squaredNmsRadius;
  });
}

function getInstanceScore(
    existingPoses: Pose[], squaredNmsRadius: number,
    instanceKeypoints: Keypoint[]) {
  let notOverlappedKeypointScores =
      instanceKeypoints.reduce((result, {point, score}, keypointId): number => {
        if (!withinNmsRadiusOfCorrespondingPoint(
                existingPoses, squaredNmsRadius, point, keypointId)) {
          result += score;
        }
        return result;
      }, 0.0);

  return notOverlappedKeypointScores /= instanceKeypoints.length;
}

async function toTensorBuffer3D(tensor: tf.Tensor3D) {
  const tensorData = await tensor.data();

  return new tf.TensorBuffer<tf.Rank.R3>(tensor.shape, 'float32', tensorData);
}

async function toTensorBuffers3D(tensors: tf.Tensor3D[]) {
  return Promise.all(tensors.map(toTensorBuffer3D));
}

// A point (y, x) is considered as root part candidate if its score is a
// maximum in a window |y - y'| <= kLocalMaximumRadius, |x - x'| <=
// kLocalMaximumRadius.
const kLocalMaximumRadius = 1;

export async function decode(
    heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D,
    displacementsFwd: tf.Tensor3D, displacementsBwd: tf.Tensor3D,
    outputStride: number, maxDetections: number, scoreThreshold = 0.5,
    nmsRadius = 20) {
  const poses: Pose[] = [];

  const
      [scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
       displacementsBwdBuffer] =
          await toTensorBuffers3D(
              [heatmapScores, offsets, displacementsFwd, displacementsBwd]);

  // const queueStartTime = new Date().getTime();
  const queue = buildPartWithScoreQueue(
      scoreThreshold, kLocalMaximumRadius, scoresBuffer);
  // console.log('queue build time', new Date().getTime() - queueStartTime);

  const squaredNmsRadius = nmsRadius * nmsRadius;

  // Generate at most max_detections object instances per image in
  // decreasing root part score order.
  while (poses.length < maxDetections && !queue.empty()) {
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

    // const decodeStartTime = new Date().getTime();
    const keypoints = decodePose(
        root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer,
        displacementsBwdBuffer);

    const score = getInstanceScore(poses, squaredNmsRadius, keypoints);

    poses.push({keypoints, score});
  }

  return poses;
}
