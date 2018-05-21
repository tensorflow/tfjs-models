import * as tf from '@tensorflow/tfjs';
export declare type Vector2D = {
    y: number;
    x: number;
};
export declare type Part = {
    heatmapX: number;
    heatmapY: number;
    id: number;
};
export declare type PartWithScore = {
    score: number;
    part: Part;
};
export declare type Keypoint = {
    score: number;
    position: Vector2D;
    part: string;
};
export declare type Pose = {
    keypoints: Keypoint[];
    score: number;
};
export declare type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;
