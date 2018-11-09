import * as tf from '@tensorflow/tfjs';

export type BodyPixInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;
