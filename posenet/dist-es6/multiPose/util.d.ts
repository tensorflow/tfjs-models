import { Part, TensorBuffer3D, Vector2D } from '../types';
export declare function getOffsetPoint(y: number, x: number, keypoint: number, offsets: TensorBuffer3D): Vector2D;
export declare function getImageCoords(part: Part, outputStride: number, offsets: TensorBuffer3D): Vector2D;
export declare function fillArray<T>(element: T, size: number): T[];
export declare function clamp(a: number, min: number, max: number): number;
export declare function squaredDistance(y1: number, x1: number, y2: number, x2: number): number;
export declare function addVectors(a: Vector2D, b: Vector2D): Vector2D;
export declare function clampVector(a: Vector2D, min: number, max: number): Vector2D;
