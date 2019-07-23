import * as tf from '@tensorflow/tfjs-core';

export type BodyPixInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;


export type PartSegmentation = {
  data: Int32Array,
  width: number,
  height: number
};

export declare interface Padding {
  top: number, bottom: number, left: number, right: number
}

export declare type Part = {
  heatmapX: number,
  heatmapY: number,
  id: number
};

export declare type Vector2D = {
  y: number,
  x: number
};

export type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;

export declare type PartWithScore = {
  score: number,
  part: Part
};

export declare type Keypoint = {
  score: number,
  position: Vector2D,
  part: string
};

export declare type Pose = {
  keypoints: Keypoint[],
  score: number,
};

export type PersonSegmentation = {
  data: Uint8Array,
  width: number,
  height: number
    data2?: Float32Array,
    data3?: Float32Array,
    data4?: Float32Array,
    poses?: Pose[]
  };
