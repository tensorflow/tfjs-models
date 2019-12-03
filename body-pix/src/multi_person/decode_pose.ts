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

import {NumberTuple, PART_IDS, PART_NAMES, POSE_CHAIN} from '../keypoints';
import {Keypoint, PartWithScore, TensorBuffer3D, Vector2D} from '../types';

import {clamp, getOffsetPoint} from './util';
import {addVectors, getImageCoords} from './util';

const parentChildrenTuples: NumberTuple[] = POSE_CHAIN.map(
    ([parentJoinName, childJoinName]): NumberTuple =>
        ([PART_IDS[parentJoinName], PART_IDS[childJoinName]]));

const parentToChildEdges: number[] =
    parentChildrenTuples.map(([, childJointId]) => childJointId);

const childToParentEdges: number[] =
    parentChildrenTuples.map(([
                               parentJointId,
                             ]) => parentJointId);

function getDisplacement(
    edgeId: number, point: Vector2D, displacements: TensorBuffer3D): Vector2D {
  const numEdges = displacements.shape[2] / 2;
  return {
    y: displacements.get(point.y, point.x, edgeId),
    x: displacements.get(point.y, point.x, numEdges + edgeId)
  };
}

function getStridedIndexNearPoint(
    point: Vector2D, outputStride: number, height: number,
    width: number): Vector2D {
  return {
    y: clamp(Math.round(point.y / outputStride), 0, height - 1),
    x: clamp(Math.round(point.x / outputStride), 0, width - 1)
  };
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
    scoresBuffer: TensorBuffer3D, offsets: TensorBuffer3D, outputStride: number,
    displacements: TensorBuffer3D, offsetRefineStep = 2): Keypoint {
  const [height, width] = scoresBuffer.shape;

  // Nearest neighbor interpolation for the source->target displacements.
  const sourceKeypointIndices = getStridedIndexNearPoint(
      sourceKeypoint.position, outputStride, height, width);

  const displacement =
      getDisplacement(edgeId, sourceKeypointIndices, displacements);

  const displacedPoint = addVectors(sourceKeypoint.position, displacement);
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

  return {position: targetKeypoint, part: PART_NAMES[targetKeypointId], score};
}

/**
 * Follows the displacement fields to decode the full pose of the object
 * instance given the position of a part that acts as root.
 *
 * @return An array of decoded keypoints and their scores for a single pose
 */
export function decodePose(
    root: PartWithScore, scores: TensorBuffer3D, offsets: TensorBuffer3D,
    outputStride: number, displacementsFwd: TensorBuffer3D,
    displacementsBwd: TensorBuffer3D): Keypoint[] {
  const numParts = scores.shape[2];
  const numEdges = parentToChildEdges.length;

  const instanceKeypoints: Keypoint[] = new Array(numParts);
  // Start a new detection instance at the position of the root.
  const {part: rootPart, score: rootScore} = root;
  const rootPoint = getImageCoords(rootPart, outputStride, offsets);

  instanceKeypoints[rootPart.id] = {
    score: rootScore,
    part: PART_NAMES[rootPart.id],
    position: rootPoint
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
