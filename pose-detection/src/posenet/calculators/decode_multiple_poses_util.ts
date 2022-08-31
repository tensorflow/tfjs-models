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

import {COCO_KEYPOINTS} from '../../constants';
import {Keypoint} from '../../shared/calculators/interfaces/common_interfaces';
import {Pose} from '../../types';
import {NUM_KEYPOINTS, POSE_CHAIN} from '../constants';
import {NumberDict, NumberTuple, Part, PartWithScore, Vector2D} from '../types';

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<Array<tf.TensorBuffer<tf.Rank.R3>>> {
  return Promise.all(tensors.map(tensor => tensor.buffer()));
}

export function getOffsetPoint(
    y: number, x: number, keypoint: number,
    offsets: tf.TensorBuffer<tf.Rank.R3>): Vector2D {
  return {
    y: offsets.get(y, x, keypoint),
    x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}

export function getImageCoords(
    part: Part, outputStride: number,
    offsets: tf.TensorBuffer<tf.Rank.R3>): Vector2D {
  const {heatmapY, heatmapX, id: keypoint} = part;
  const {y, x} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets);
  return {
    x: part.heatmapX * outputStride + x,
    y: part.heatmapY * outputStride + y
  };
}

export function squaredDistance(
    y1: number, x1: number, y2: number, x2: number): number {
  const dy = y2 - y1;
  const dx = x2 - x1;
  return dy * dy + dx * dx;
}

export function withinNmsRadiusOfCorrespondingPoint(
    poses: Pose[], squaredNmsRadius: number, {x, y}: {x: number, y: number},
    keypointId: number): boolean {
  return poses.some(({keypoints}) => {
    return squaredDistance(
               y, x, keypoints[keypointId].y, keypoints[keypointId].x) <=
        squaredNmsRadius;
  });
}

const partIds =
    // tslint:disable-next-line: no-unnecessary-type-assertion
    COCO_KEYPOINTS.reduce((result: NumberDict, jointName, i): NumberDict => {
      result[jointName] = i;
      return result;
    }, {}) as NumberDict;

const parentChildrenTuples: NumberTuple[] = POSE_CHAIN.map(
    ([parentJoinName, childJoinName]): NumberTuple =>
        ([partIds[parentJoinName], partIds[childJoinName]]));
const parentToChildEdges: number[] =
    parentChildrenTuples.map(([, childJointId]) => childJointId);

const childToParentEdges: number[] =
    parentChildrenTuples.map(([
                               parentJointId,
                             ]) => parentJointId);

function clamp(a: number, min: number, max: number): number {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

function getStridedIndexNearPoint(
    point: Vector2D, outputStride: number, height: number,
    width: number): Vector2D {
  return {
    y: clamp(Math.round(point.y / outputStride), 0, height - 1),
    x: clamp(Math.round(point.x / outputStride), 0, width - 1)
  };
}

function getDisplacement(
    edgeId: number, point: Vector2D,
    displacements: tf.TensorBuffer<tf.Rank.R3>): Vector2D {
  const numEdges = displacements.shape[2] / 2;
  return {
    y: displacements.get(point.y, point.x, edgeId),
    x: displacements.get(point.y, point.x, numEdges + edgeId)
  };
}

export function addVectors(a: Vector2D, b: Vector2D): Vector2D {
  return {x: a.x + b.x, y: a.y + b.y};
}

/**
 * We get a new keypoint along the `edgeId` for the pose instance, assuming
 * that the position of the `idSource` part is already known. For this, we
 * follow the displacement vector from the source to target part (stored in
 * the `i`-t channel of the displacement tensor). The displaced keypoint
 * vector is refined using the offset vector by `offsetRefineStep` times.
 */
function traverseToTargetKeypoint(
    edgeId: number, sourceKeypoint: Keypoint, targetKeypointId: number,
    scoresBuffer: tf.TensorBuffer<tf.Rank.R3>,
    offsets: tf.TensorBuffer<tf.Rank.R3>, outputStride: number,
    displacements: tf.TensorBuffer<tf.Rank.R3>,
    offsetRefineStep = 2): Keypoint {
  const [height, width] = scoresBuffer.shape;

  const point = {y: sourceKeypoint.y, x: sourceKeypoint.x};

  // Nearest neighbor interpolation for the source->target displacements.
  const sourceKeypointIndices =
      getStridedIndexNearPoint(point, outputStride, height, width);

  const displacement =
      getDisplacement(edgeId, sourceKeypointIndices, displacements);

  const displacedPoint = addVectors(point, displacement);
  let targetKeypoint = displacedPoint;
  for (let i = 0; i < offsetRefineStep; i++) {
    const targetKeypointIndices =
        getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);

    const offsetPoint = getOffsetPoint(
        targetKeypointIndices.y, targetKeypointIndices.x, targetKeypointId,
        offsets);

    targetKeypoint = addVectors(
        {
          x: targetKeypointIndices.x * outputStride,
          y: targetKeypointIndices.y * outputStride
        },
        {x: offsetPoint.x, y: offsetPoint.y});
  }
  const targetKeyPointIndices =
      getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);
  const score = scoresBuffer.get(
      targetKeyPointIndices.y, targetKeyPointIndices.x, targetKeypointId);

  return {
    y: targetKeypoint.y,
    x: targetKeypoint.x,
    name: COCO_KEYPOINTS[targetKeypointId],
    score
  };
}
/**
 * Follows the displacement fields to decode the full pose of the object
 * instance given the position of a part that acts as root.
 *
 * @return An array of decoded keypoints and their scores for a single pose
 */
export function decodePose(
    root: PartWithScore, scores: tf.TensorBuffer<tf.Rank.R3>,
    offsets: tf.TensorBuffer<tf.Rank.R3>, outputStride: number,
    displacementsFwd: tf.TensorBuffer<tf.Rank.R3>,
    displacementsBwd: tf.TensorBuffer<tf.Rank.R3>): Keypoint[] {
  const numParts = scores.shape[2];
  const numEdges = parentToChildEdges.length;

  const instanceKeypoints: Keypoint[] = new Array(numParts);
  // Start a new detection instance at the position of the root.
  const {part: rootPart, score: rootScore} = root;
  const rootPoint = getImageCoords(rootPart, outputStride, offsets);

  instanceKeypoints[rootPart.id] = {
    score: rootScore,
    name: COCO_KEYPOINTS[rootPart.id],
    y: rootPoint.y,
    x: rootPoint.x
  };

  // Decode the part positions upwards in the tree, following the backward
  // displacements.
  for (let edge = numEdges - 1; edge >= 0; --edge) {
    const sourceKeypointId = parentToChildEdges[edge];
    const targetKeypointId = childToParentEdges[edge];
    if (instanceKeypoints[sourceKeypointId] &&
        !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
          edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
          offsets, outputStride, displacementsBwd);
    }
  }

  // Decode the part positions downwards in the tree, following the forward
  // displacements.
  for (let edge = 0; edge < numEdges; ++edge) {
    const sourceKeypointId = childToParentEdges[edge];
    const targetKeypointId = parentToChildEdges[edge];
    if (instanceKeypoints[sourceKeypointId] &&
        !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
          edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
          offsets, outputStride, displacementsFwd);
    }
  }

  return instanceKeypoints;
}

/* Score the newly proposed object instance without taking into account
 * the scores of the parts that overlap with any previously detected
 * instance.
 */
export function getInstanceScore(
    existingPoses: Pose[], squaredNmsRadius: number,
    instanceKeypoints: Keypoint[]): number {
  let notOverlappedKeypointScores =
      instanceKeypoints.reduce((result, {y, x, score}, keypointId): number => {
        if (!withinNmsRadiusOfCorrespondingPoint(
                existingPoses, squaredNmsRadius, {y, x}, keypointId)) {
          result += score;
        }
        return result;
      }, 0.0);

  return notOverlappedKeypointScores /= instanceKeypoints.length;
}
