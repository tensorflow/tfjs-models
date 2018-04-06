import * as tf from '@tensorflow/tfjs-core';

export type Vector2D = {
  y: number,
  x: number
};

export type Part = {
  heatmapX: number,
  heatmapY: number,
  id: number
};

export type PartWithScore = {
  score: number,
  part: Part
};

export type Keypoint = {
  score: number,
  point: Vector2D
};

export type Pose = {
  keypoints: Keypoint[],
  score: number
};

export type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;
