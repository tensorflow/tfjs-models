import * as tf from '@tensorflow/tfjs';

export type PersonSegmentationInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;
