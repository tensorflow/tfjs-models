import * as tf from '@tensorflow/tfjs';
export declare function getPointsConfidence(heatmapScores: tf.TensorBuffer<tf.Rank.R3>, heatMapCoords: tf.TensorBuffer<tf.Rank.R2>): Float32Array;
export declare function getOffsetVectors(heatMapCoordsBuffer: tf.TensorBuffer<tf.Rank.R2>, offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>): tf.Tensor2D;
export declare function getOffsetPoints(heatMapCoordsBuffer: tf.TensorBuffer<tf.Rank.R2>, outputStride: number, offsetsBuffer: tf.TensorBuffer<tf.Rank.R3>): tf.Tensor2D;
