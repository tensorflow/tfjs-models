/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

import {config} from './config';
import {connectedComponents} from './connectedComponents';
import {Point} from './geometry';
import {minAreaRect} from './minAreaRect';
import {Queue} from './queue';
import {Box, QuantizationBytes, TextDetectionInput} from './types';

export const getURL = (quantizationBytes: QuantizationBytes) => {
  return `${config['BASE_PATH']}/${
      quantizationBytes ? `quantized/${quantizationBytes}/` :
                          ''}psenet/model.json`;
};

export const progressiveScaleExpansion =
    (kernels: tf.Tensor2D[],
     minAreaThreshold = config['MINIMAL_AREA_THRESHOLD']) => {
      const [height, width] = kernels[0].shape;
      const lastSegmentationMapData =
          Array.from(kernels[kernels.length - 1].arraySync());
      const {labelsCount, labels} =
          connectedComponents(lastSegmentationMapData);
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
            if (areaSizes[label] < minAreaThreshold) {
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

export const detect =
    (segmentationMaps: tf.Tensor3D, originalHeight: number,
     originalWidth: number, minAreaThreshold = config['MINIMAL_AREA_THRESHOLD'],
     segmentationMapThreshold = config['SEGMENTATION_MAP_THRESHOLD'],
     clippingEdge = config['MAX_SIDE_LENGTH']): Box[] => {
      const [height, width, mapCount] = segmentationMaps.shape;
      const segmentationMapsData = segmentationMaps.arraySync();
      tf.dispose(segmentationMaps);
      const one = tf.ones([height, width], 'int32');
      const zero = tf.zeros([height, width], 'int32');
      const threshold = tf.fill([height, width], segmentationMapThreshold);
      const kernels = new Array<tf.Tensor2D>();
      for (let mapIdx = mapCount - 1; mapIdx > -1; --mapIdx) {
        const segmentationMapBuffer = tf.buffer([height, width], 'int32');
        for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
            segmentationMapBuffer.set(
                segmentationMapsData[rowIdx][columnIdx][mapIdx], rowIdx,
                columnIdx);
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
      const [heightScalingFactor, widthScalingFactor] =
          computeScalingFactors(originalHeight, originalWidth, clippingEdge);
      if (kernels.length > 0) {
        const {segmentationMapBuffer, recognizedLabels} =
            progressiveScaleExpansion(kernels, minAreaThreshold);
        tf.dispose(kernels);
        const targetHeight = Math.round(originalHeight * heightScalingFactor);
        const targetWidth = Math.round(originalWidth * widthScalingFactor);
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
        const points: {[label: number]: Point[]} = {};
        for (let rowIdx = 0; rowIdx < targetHeight; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < targetWidth; ++columnIdx) {
            const label = resizedSegmentationMapData[rowIdx][columnIdx];
            if (recognizedLabels.has(label)) {
              if (!points[label]) {
                points[label] = [];
              }
              points[label].push(new Point(columnIdx, rowIdx));
            }
          }
        }
        const boxes: Box[] = [];
        const clip = (size: number, edge: number) =>
            (size > edge ? edge : size);
        Object.keys(points).forEach((labelStr) => {
          const label = Number(labelStr);
          const box = minAreaRect(points[label]);
          for (let pointIdx = 0; pointIdx < box.length; ++pointIdx) {
            const point = box[pointIdx];
            const scaledX = clip(point.x / widthScalingFactor, originalWidth);
            const scaledY = clip(point.y / heightScalingFactor, originalHeight);
            box[pointIdx] = new Point(scaledX, scaledY);
          }
          boxes.push(box);
        });
        return boxes;
      }
      return [];
    };

export const computeScalingFactors =
    (height: number, width: number, clippingEdge: number): [number, number] => {
      const maxSide = Math.max(width, height);
      const ratio = maxSide > clippingEdge ? clippingEdge / maxSide : 1;

      const getScalingFactor = (side: number) => {
        const roundedSide = Math.round(side * ratio);
        return (roundedSide % 32 === 0 ?
                    roundedSide :
                    (Math.floor(roundedSide / 32) + 1) * 32) /
            side;
      };

      const heightScalingRatio = getScalingFactor(height);
      const widthScalingRatio = getScalingFactor(width);
      return [heightScalingRatio, widthScalingRatio];
    };

export const cropAndResize = (input: TextDetectionInput,
                              clippingEdge =
                                  config['MAX_SIDE_LENGTH']): tf.Tensor3D => {
  return tf.tidy(() => {
    const image: tf.Tensor3D = (input instanceof tf.Tensor ?
                                    input :
                                    tf.browser.fromPixels(
                                        input as ImageData | HTMLImageElement |
                                        HTMLCanvasElement | HTMLVideoElement))
                                   .toFloat();

    const [height, width] = image.shape;
    const [heightScalingFactor, widthScalingFactor] =
        computeScalingFactors(height, width, clippingEdge);
    const targetHeight = Math.round(height * heightScalingFactor);
    const targetWidth = Math.round(width * widthScalingFactor);
    const processedImage =
        tf.image.resizeBilinear(image, [targetHeight, targetWidth]);

    return processedImage;
  });
};
