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
import {Point} from './geometry';
import {minAreaRect} from './minAreaRect';
import {progressiveScaleExpansion} from './progressiveScaleExpansion';
import {Box, QuantizationBytes, TextDetectionInput} from './types';

export const getURL = (quantizationBytes: QuantizationBytes) => {
  return `${config['BASE_PATH']}/${
      quantizationBytes ? `quantized/${quantizationBytes}/` :
                          ''}psenet/model.json`;
};

export const detect =
    (kernelScores: tf.Tensor3D, originalHeight: number, originalWidth: number,
     minKernelArea = config['MIN_KERNEL_AREA'], minScore = config['MIN_SCORE'],
     maxSideLength = config['MAX_SIDE_LENGTH']): Box[] => {
      const [height, width, numOfKernels] = kernelScores.shape;
      const kernelScoreData = kernelScores.arraySync();
      tf.dispose(kernelScores);
      const one = tf.ones([height, width], 'int32');
      const zero = tf.zeros([height, width], 'int32');
      const threshold = tf.fill([height, width], minScore);
      const kernels = new Array<tf.Tensor2D>();
      for (let kernelIdx = numOfKernels - 1; kernelIdx > -1; --kernelIdx) {
        const kernelScoreBuffer = tf.buffer([height, width], 'int32');
        for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
            kernelScoreBuffer.set(
                kernelScoreData[rowIdx][columnIdx][kernelIdx], rowIdx,
                columnIdx);
          }
        }
        const kernelScore = kernelScoreBuffer.toTensor();
        const kernel = tf.tidy(
            () => tf.where(kernelScore.greater(threshold), one, zero) as
                tf.Tensor2D);
        kernels.push(kernel);
        tf.dispose(kernelScore);
      }
      tf.dispose(one);
      tf.dispose(zero);
      tf.dispose(threshold);
      const [heightScalingFactor, widthScalingFactor] =
          computeScalingFactors(originalHeight, originalWidth, maxSideLength);
      if (kernels.length > 0) {
        const {segmentationMapBuffer, recognizedLabels} =
            progressiveScaleExpansion(kernels, minKernelArea);
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
