import {jointIdsByName, NumberTuple, StringTuple} from '../keypoints';
import {Keypoint, PartWithScore, TensorBuffer3D, Vector2D} from '../types';
import {clamp, getOffsetPoint} from './util';

import {addVectors, getImageCoords} from './util';

/*
 * Define the skeleton. This defines the parent->child relationships of our
 * tree. Arbitrarily this defines the nose as the root of the tree, however
 * since we will infer the displacement for both parent->child and
 * child->parent, we can define the tree root as any node.
 */
const poseChain: StringTuple[] = [
  ['nose', 'left_eye'], ['left_eye', 'left_ear'], ['nose', 'right_eye'],
  ['right_eye', 'right_ear'], ['nose', 'left_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['left_shoulder', 'left_hip'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['nose', 'right_shoulder'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['right_shoulder', 'right_hip'], ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle']
];

const parentChildrenTuples: NumberTuple[] = poseChain.map(
    ([parentJoinName, childJoinName]): NumberTuple =>
        ([jointIdsByName[parentJoinName], jointIdsByName[childJoinName]]));

const parentToChildEdges: number[] =
    parentChildrenTuples.map(([, childJointId]) => childJointId);

const childToParentEdges: number[] =
    parentChildrenTuples.map(([
                               parentJointId,
                             ]) => parentJointId);

function getDisplacement(
    i: number, point: Vector2D, displacements: TensorBuffer3D): Vector2D {
  const numEdges = displacements.shape[2] / 2;
  return {
    y: displacements.get(point.y, point.x, i),
    x: displacements.get(point.y, point.x, numEdges + i)
  };
}

function decode(
    point: Vector2D, outputStride: number, height: number,
    width: number): Vector2D {
  return {
    y: clamp(Math.round(point.y / outputStride), 0, height - 1),
    x: clamp(Math.round(point.x / outputStride), 0, width - 1)
  };
}

// We add a new `id_target` part to the detection instance, assuming that the
// position of the `id_source` part is already known. For this, we follow the
// displacement vector from the source to target part (stored in the `i`-th
// channel of the displacement tensor), followed by a local search in a small
// window in order to find the position of maximum score for the `id_target`
// part.
function traverseToTargetKeypoint(
    edgeId: number, sourceKeypoint: Keypoint, targetKeypointId: number,
    scoresBuffer: TensorBuffer3D, offsets: TensorBuffer3D, outputStride: number,
    displacements: TensorBuffer3D): Keypoint {
  const [height, width] = scoresBuffer.shape;

  // Nearest neighbor interpolation for the source->target displacements.
  const sourceKeypointIndeces =
      decode(sourceKeypoint.point, outputStride, height, width);

  const displacement =
      getDisplacement(edgeId, sourceKeypointIndeces, displacements);

  const displacedPoint = addVectors(sourceKeypoint.point, displacement);

  const displacedPointIndeces =
      decode(displacedPoint, outputStride, height, width);

  const offsetPoint = getOffsetPoint(
      displacedPointIndeces.y, displacedPointIndeces.x, targetKeypointId,
      offsets);

  const targetKeypoint =
      addVectors(displacedPoint, {x: offsetPoint.x, y: offsetPoint.y});

  const targetKeypointIndeces =
      decode(targetKeypoint, outputStride, height, width);

  const score = scoresBuffer.get(
      targetKeypointIndeces.y, targetKeypointIndeces.x, targetKeypointId);

  return {point: targetKeypoint, score};
}

export function decodePose(
    root: PartWithScore, scores: TensorBuffer3D, offsets: TensorBuffer3D,
    outputStride: number, displacementsFwd: TensorBuffer3D,
    displacementsBwd: TensorBuffer3D) {
  // Start a new detection instance at the position of the root.
  const numParts = scores.shape[2];
  const numEdges = parentToChildEdges.length;

  const instanceKeypoints: Keypoint[] = new Array(numParts);
  const {part: rootPart, score: rootScore} = root;
  const rootPoint = getImageCoords(rootPart, outputStride, offsets);

  instanceKeypoints[rootPart.id] = {score: rootScore, point: rootPoint};

  // console.log('root part', rootPart, rootPoint);

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
