import * as tf from '@tensorflow/tfjs-core';
import {config} from './config';
import {connectedComponents} from './connectedComponents';
import {Queue} from './queue';
export const progressiveScaleExpansion =
    (kernels: tf.Tensor2D[], minKernelArea = config['MIN_KERNEL_AREA']) => {
      const [height, width] = kernels[0].shape;
      const lastKernel = kernels[kernels.length - 1];
      const lastKernelData = lastKernel.arraySync();
      tf.dispose(lastKernel);
      const {labelsCount, labels} = connectedComponents(lastKernelData);
      const areaSizes = Array<number>(labelsCount);
      for (let rowIdx = 0; rowIdx < labels.length; rowIdx++) {
        for (let colIdx = 0; colIdx < labels[0].length; colIdx++) {
          const label = labels[rowIdx][colIdx];
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
      for (let rowIdx = 0; rowIdx < labels.length; rowIdx++) {
        for (let colIdx = 0; colIdx < labels[0].length; colIdx++) {
          const label = labels[rowIdx][colIdx];
          if (label > 0) {
            if (areaSizes[label] < minKernelArea) {
              labels[rowIdx][colIdx] = 0;
            } else {
              queues[currentQueueIdx].push([colIdx, rowIdx, label]);
              segmentationMapBuffer.set(label, rowIdx, colIdx);
              recognizedLabels.add(label);
            }
          }
        }
      }
      const dx = [-1, 1, 0, 0];
      const dy = [0, 0, -1, 1];
      for (let kernelIdx = kernels.length - 2; kernelIdx > -1; --kernelIdx) {
        const kernel = kernels[kernelIdx];
        const kernelData = kernel.arraySync();
        while (!queues[currentQueueIdx].empty()) {
          const [xCoordinate, yCoordinate, label] =
              queues[currentQueueIdx].pop();
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
