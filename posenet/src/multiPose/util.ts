import {NUM_KEYPOINTS} from '../keypoints';
import {Part, TensorBuffer3D, Vector2D} from '../types';

export function getOffsetPoint(
    y: number, x: number, keypoint: number, offsets: TensorBuffer3D) {
  return {
    y: offsets.get(y, x, keypoint),
    x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}

export function getImageCoords(
    part: Part, outputStride: number, offsets: TensorBuffer3D) {
  const {heatmapY, heatmapX, id: keypoint} = part;
  const {y, x} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets);
  // console.log('offset:', offsetY, offsetX);
  return {
    x: part.heatmapX * outputStride + x,
    y: part.heatmapY * outputStride + y
  };
}

export function fillArray<T>(element: T, size: number) {
  const result: T[] = new Array(size);

  for (let i = 0; i < size; i++) {
    result[i] = element;
  }

  return result;
}

export function clamp(a: number, min: number, max: number) {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

export function squaredDistance(
    y1: number, x1: number, y2: number, x2: number) {
  const dy = y2 - y1;
  const dx = x2 - x1;
  return dy * dy + dx * dx;
}

export function addVectors(a: Vector2D, b: Vector2D): Vector2D {
  return {x: a.x + b.x, y: a.y + b.y};
}

export function clampVector(a: Vector2D, min: number, max: number) {
  return {y: clamp(a.y, min, max), x: clamp(a.x, min, max)};
}
