import * as tf from '@tensorflow/tfjs-core';

import {Keypoint, Pose} from '../types';

import {argmax2d} from './argmax2d';
import {getOffsetPoints, getPointsConfidence} from './util';

export function decode(
    heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D,
    outputStride: number): Pose {
  let totalScore = 0.0;

  const heatmapValues = argmax2d(heatmapScores);
  const offsetPoints =
      getOffsetPoints(heatmapValues, outputStride, offsets).buffer();

  const keypointConfidence =
      Array.from(getPointsConfidence(heatmapScores, heatmapValues));

  const keypoints = keypointConfidence.map((score, keypointId): Keypoint => {
    totalScore += score;
    return {
      point: {
        y: offsetPoints.get(keypointId, 0),
        x: offsetPoints.get(keypointId, 1)
      },
      score
    };
  });

  return {keypoints, score: totalScore / keypoints.length};
}
