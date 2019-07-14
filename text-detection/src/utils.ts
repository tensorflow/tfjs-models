import * as tf from '@tensorflow/tfjs';

import {config} from './config';
import cv from './opencv';
import {TextDetectionInput} from './types';

export class Queue<T> {
  _store: T[] = [];
  push(val: T) {
    this._store.push(val);
  }
  pop(): T|undefined {
    return this._store.shift();
  }
}

export const progressiveScaleExpansion = (kernels: tf.Tensor2D[]) => {
  const [height, width] = kernels[0].shape;
  // const predictionsBuffer = tf.buffer([height, width], 'int32');
  const lastSegmentationMapData =
      Array.from(kernels[kernels.length - 1].dataSync());
  const lastSegmentationMapMatrix =
      cv.matFromArray(height, width, cv.CV_8U, lastSegmentationMapData);
  const labels = new cv.Mat();
  const labelsCount =
      cv.connectedComponents(lastSegmentationMapMatrix, labels, 4);
  const areaSizes = Array<number>(labelsCount);
  for (let rowIdx = 0; rowIdx < labels.rows; rowIdx++) {
    for (let colIdx = 0; colIdx < labels.cols; colIdx++) {
      const label = labels.ucharPtr(rowIdx, colIdx)[0];
      if (label > 0) {
        areaSizes[label] += 1;
      }
    }
  }
  const labelsToIgnore = new Set<number>();
  const points: number[][] = [];  // An array of [x, y]-coordinates
  for (let rowIdx = 0; rowIdx < labels.rows; rowIdx++) {
    for (let colIdx = 0; colIdx < labels.cols; colIdx++) {
      const label = labels.ucharPtr(rowIdx, colIdx)[0];
      if (labelsToIgnore.has(label)) {
        labels.ucharPtr(rowIdx, colIdx)[0] = 0;
      } else if (label > 0) {
        points.push([colIdx, rowIdx]);
      }
    }
  }
  const queues = Array<Queue<[number, number, number]>>(2);
  let currentQueueIdx = 0;
  const dx = [-1, 1, 0, 0];
  const dy = [0, 0, -1, 1];
};

// kernelQueue = queue.Queue(maxsize=0)
// next_queue = queue.Queue(maxsize=0)
// points = np.array(np.where(label > 0)).transpose((1, 0))

// for point_idx in range(points.shape[0]):
//     x, y = points[point_idx, 0], points[point_idx, 1]
//     l = label[x, y]
//     kernelQueue.put((x, y, l))
//     pred[x, y] = l

// dx = [-1, 1, 0, 0]
// dy = [0, 0, -1, 1]
// for kernal_idx in range(kernal_num - 2, -1, -1):
//     kernal = kernals[kernal_idx].copy()
//     while not kernelQueue.empty():
//         (x, y, l) = kernelQueue.get()

//         is_edge = True
//         for j in range(4):
//             tmpx = x + dx[j]
//             tmpy = y + dy[j]
//             if (
//                 tmpx < 0
//                 or tmpx >= kernal.shape[0]
//                 or tmpy < 0
//                 or tmpy >= kernal.shape[1]
//             ):
//                 continue
//             if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
//                 continue

//             kernelQueue.put((tmpx, tmpy, l))
//             pred[tmpx, tmpy] = l
//             is_edge = False
//         if is_edge:
//             next_queue.put((x, y, l))

//     # kernal[pred > 0] = 0
//     kernelQueue, next_queue = next_queue, kernelQueue

//     # points = np.array(np.where(pred > 0)).transpose((1, 0))
//     # for point_idx in range(points.shape[0]):
//     #     x, y = points[point_idx, 0], points[point_idx, 1]
//     #     l = pred[x, y]
//     #     queue.put((x, y, l))

// return pred, label_values

export const detect = (segmentationMaps: tf.Tensor3D) => {
  const [height, width, mapCount] = segmentationMaps.shape;
  const segmentationMapsData = segmentationMaps.arraySync();
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
    const kernel =
        tf.where(segmentationMap.greater(threshold), one, zero) as tf.Tensor2D;
    kernels.push(kernel);
  }
  if (kernels.length > 0) {
    progressiveScaleExpansion(kernels);
  }
  // mask_res, label_values = pse(kernals, min_area_thresh)
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
    const processedImage =
        tf.image.resizeBilinear(image, [height, width].map((side, idx) => {
          return Math.round(side * resizeRatios[idx]);
        }) as [number, number]);

    return processedImage;
  });
}
