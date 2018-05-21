import { Keypoint, PartWithScore, TensorBuffer3D } from '../types';
export declare function decodePose(root: PartWithScore, scores: TensorBuffer3D, offsets: TensorBuffer3D, outputStride: number, displacementsFwd: TensorBuffer3D, displacementsBwd: TensorBuffer3D): Keypoint[];
