import * as tf from '@tensorflow/tfjs';

import {config} from './config';
import cv from './opencv';
import {TextDetectionInput} from './types';
export class Queue<T> {
  private _store: T[] = [];
  public push(val: T) {
    this._store.push(val);
  }
  public pop(): T|undefined {
    return this._store.shift();
  }
  public empty() {
    return this._store.length === 0;
  }
  public size() {
    return this._store.length;
  }
}

export const progressiveScaleExpansion = (kernels: tf.Tensor2D[]) => {
  const [height, width] = kernels[0].shape;
  const lastSegmentationMapData =
      Array.from(kernels[kernels.length - 1].dataSync());
  const lastSegmentationMapMatrix =
      cv.matFromArray(height, width, cv.CV_8U, lastSegmentationMapData);
  const labels = new cv.Mat();
  const labelsCount =
      cv.connectedComponents(lastSegmentationMapMatrix, labels, 4);
  lastSegmentationMapMatrix.delete();
  const areaSizes = Array<number>(labelsCount);
  for (let rowIdx = 0; rowIdx < labels.rows; rowIdx++) {
    for (let colIdx = 0; colIdx < labels.cols; colIdx++) {
      const label = labels.ucharPtr(rowIdx, colIdx)[0];
      if (label > 0) {
        areaSizes[label] += 1;
      }
    }
  }
  const recognizedLabels = new Set<number>();
  const queues: Array<Queue<[number, number, number]>> = [
    new Queue<[number, number, number]>(),
    new Queue<[number, number, number]>()
  ];
  let currentQueueIdx = 0;
  const segmentationMapBuffer = tf.buffer([height, width], 'int32');
  for (let rowIdx = 0; rowIdx < labels.rows; rowIdx++) {
    for (let colIdx = 0; colIdx < labels.cols; colIdx++) {
      const label = labels.ucharPtr(rowIdx, colIdx)[0];
      if (label > 0) {
        if (areaSizes[label] < config['MINIMAL_AREA_THRESHOLD']) {
          labels.ucharPtr(rowIdx, colIdx)[0] = 0;
        } else {
          queues[currentQueueIdx].push([colIdx, rowIdx, label]);
          segmentationMapBuffer.set(label, rowIdx, colIdx);
          recognizedLabels.add(label);
        }
      }
    }
  }
  labels.delete();
  const dx = [-1, 1, 0, 0];
  const dy = [0, 0, -1, 1];
  for (let kernelIdx = kernels.length - 2; kernelIdx > -1; --kernelIdx) {
    const kernel = kernels[kernelIdx];
    const kernelData = kernel.arraySync();
    while (!queues[currentQueueIdx].empty()) {
      const [xCoordinate, yCoordinate, label] = queues[currentQueueIdx].pop();
      let isEdge = true;
      for (let direction = 0; direction < 4; ++direction) {
        const nextX = xCoordinate + dx[direction];
        const nextY = yCoordinate + dy[direction];
        if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
          continue;
        }
        if (kernelData[nextY][nextX] === 0 ||
            segmentationMapBuffer.get(nextY, nextX) > 0) {
          continue;
        }
        queues[currentQueueIdx].push([nextX, nextY, label]);
        segmentationMapBuffer.set(label, nextY, nextX);
        isEdge = false;
      }
      if (isEdge) {
        const nextQueueIdx = currentQueueIdx ^ 1;
        queues[nextQueueIdx].push([xCoordinate, yCoordinate, label]);
      }
    }
    currentQueueIdx ^= 1;
  }
  return {segmentationMapBuffer, recognizedLabels};
};

export const detect = (segmentationMaps: tf.Tensor3D): cv.Point2[][] => {
  const [height, width, mapCount] = segmentationMaps.shape;
  const segmentationMapsData = segmentationMaps.arraySync();
  tf.dispose(segmentationMaps);
  const one = tf.ones([height, width], 'int32');
  const zero = tf.zeros([height, width], 'int32');
  const threshold =
      tf.fill([height, width], config['SEGMENTATION_MAP_THRESHOLD']);
  const kernels = new Array<tf.Tensor2D>();
  for (let mapIdx = mapCount - 1; mapIdx > -1; --mapIdx) {
    const segmentationMapBuffer = tf.buffer([height, width], 'int32');
    for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
      for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
        segmentationMapBuffer.set(
            segmentationMapsData[rowIdx][columnIdx][mapIdx], rowIdx, columnIdx);
      }
    }
    const segmentationMap = segmentationMapBuffer.toTensor();
    const kernel = tf.tidy(
        () => tf.where(segmentationMap.greater(threshold), one, zero) as
            tf.Tensor2D);
    kernels.push(kernel);
    tf.dispose(segmentationMap);
  }
  tf.dispose(one);
  tf.dispose(zero);
  tf.dispose(threshold);
  if (kernels.length > 0) {
    const {segmentationMapBuffer, recognizedLabels} =
        progressiveScaleExpansion(kernels);
    tf.dispose(kernels);
    const resizeRatios = computeTargetRatios(height, width);
    const [targetHeight, targetWidth] = [height, width].map((side, idx) => {
      return Math.round(side * resizeRatios[idx]);
    });
    const resizedSegmentationMap = tf.tidy(() => {
      const processedSegmentationMap =
          segmentationMapBuffer.toTensor().expandDims(2) as tf.Tensor3D;
      return tf.image
          .resizeNearestNeighbor(
              processedSegmentationMap, [targetHeight, targetWidth])
          .squeeze([2]);
    });
    const resizedSegmentationMapData =
        resizedSegmentationMap.arraySync() as number[][];
    tf.dispose(resizedSegmentationMap);
    if (recognizedLabels.size === 0) {
      return [];
    }
    const points: {[label: number]: {'x': number[], 'y': number[]}} = {};
    for (let rowIdx = 0; rowIdx < targetHeight; ++rowIdx) {
      for (let columnIdx = 0; columnIdx < targetWidth; ++columnIdx) {
        const label = resizedSegmentationMapData[rowIdx][columnIdx];
        if (recognizedLabels.has(label)) {
          if (!points[label]) {
            points[label] = {'x': [], 'y': []};
          }
          points[label].x.push(columnIdx);
          points[label].y.push(rowIdx);
        }
      }
    }
    const boxes: cv.Point2[][] = [];
    Object.keys(points).forEach((labelStr) => {
      const label = Number(labelStr);
      const x = cv.matFromArray(
          points[label].x.length, 1, cv.CV_32S, points[label].x);
      const y = cv.matFromArray(
          points[label].y.length, 1, cv.CV_32S, points[label].y);
      const coords = new cv.MatVector();
      coords.push_back(x);
      coords.push_back(y);
      const polytope = new cv.Mat();
      cv.merge(coords, polytope);
      const box = cv.RotatedRect.points(cv.minAreaRect(polytope));
      boxes.push(box);
      x.delete();
      y.delete();
      coords.delete();
      polytope.delete();
    });
    return boxes;
  }
  return [];
};

export const computeTargetRatios =
    (height: number, width: number): [number, number] => {
      const maxSide = Math.max(width, height)
      const ratio = maxSide > config['MAX_SIDE_LENGTH'] ?
          config['MAX_SIDE_LENGTH'] / maxSide :
          1;

      const getTargetRatio = (side: number) => {
        const roundedSide = Math.round(side * ratio);
        return (roundedSide % 32 === 0 ?
                    roundedSide :
                    (Math.floor(roundedSide / 32) + 1) * 32) /
            side;
      };

      const targetHeightRatio = getTargetRatio(height);
      const targetWidthRatio = getTargetRatio(width);
      return [targetHeightRatio, targetWidthRatio];
    };

export const cropAndResize = (input: TextDetectionInput): tf.Tensor3D => {
  return tf.tidy(() => {
    const image: tf.Tensor3D =
        (input instanceof tf.Tensor ? input : tf.browser.fromPixels(input))
            .toFloat();

    const [height, width] = image.shape;
    const resizeRatios = computeTargetRatios(height, width);
    const [targetHeight, targetWidth] = [height, width].map((side, idx) => {
      return Math.round(side * resizeRatios[idx]);
    });
    const processedImage =
        tf.image.resizeBilinear(image, [targetHeight, targetWidth]);

    return processedImage;
  });
}
