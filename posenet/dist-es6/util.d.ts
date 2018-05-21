import * as tf from '@tensorflow/tfjs';
import { OutputStride } from './mobilenet';
import { Keypoint, Pose, TensorBuffer3D, Vector2D } from './types';
export declare function getAdjacentKeyPoints(keypoints: Keypoint[], minConfidence: number): Keypoint[][];
export declare function getBoundingBox(keypoints: Keypoint[]): {
    maxX: number;
    maxY: number;
    minX: number;
    minY: number;
};
export declare function getBoundingBoxPoints(keypoints: Keypoint[]): Vector2D[];
export declare function toTensorBuffer<rank extends tf.Rank>(tensor: tf.Tensor<rank>, type?: 'float32' | 'int32'): Promise<tf.TensorBuffer<rank>>;
export declare function toTensorBuffers3D(tensors: tf.Tensor3D[]): Promise<TensorBuffer3D[]>;
export declare function scalePose(pose: Pose, scaleX: number, scaleY: number): Pose;
export declare function scalePoses(poses: Pose[], scaleY: number, scaleX: number): Pose[];
export declare function getValidResolution(imageScaleFactor: number, inputDimension: number, outputStride: OutputStride): number;
