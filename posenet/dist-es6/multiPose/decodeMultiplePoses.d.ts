import * as tf from '@tensorflow/tfjs';
import { Pose } from '../types';
export default function decodeMultiplePoses(heatmapScores: tf.Tensor3D, offsets: tf.Tensor3D, displacementsFwd: tf.Tensor3D, displacementsBwd: tf.Tensor3D, outputStride: number, maxPoseDetections: number, scoreThreshold?: number, nmsRadius?: number): Promise<Pose[]>;
