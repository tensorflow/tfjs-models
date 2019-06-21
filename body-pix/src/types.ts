import * as tf from '@tensorflow/tfjs-core';

export type BodyPixInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;

export type PersonSegmentation = {
  data: Uint8Array,
  width: number,
  height: number
};

export type PartSegmentation = {
  data: Int32Array,
  width: number,
  height: number
};
