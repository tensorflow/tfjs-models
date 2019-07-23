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
import {Box, QuantizationBytes, TextDetectionInput, TextDetectionOptions} from './types';

export const getURL = (quantizationBytes: QuantizationBytes) => {
  return `${config['BASE_PATH']}/${
      ([1, 2].indexOf(quantizationBytes) !== -1) ?
          `quantized/${quantizationBytes}/` :
          ''}psenet/model.json`;
};

export const detect = async(
    kernelScores: tf.Tensor3D, originalHeight: number, originalWidth: number,
    textDetectionOptions: TextDetectionOptions = {
      minTextBoxArea: config['MIN_TEXTBOX_AREA'],
      minConfidence: config['MIN_CONFIDENCE'],
      resizeLength: config['RESIZE_LENGTH']
    }): Promise<Box[]> => {
  if (!textDetectionOptions.minTextBoxArea) {
    textDetectionOptions.minTextBoxArea = config['MIN_TEXTBOX_AREA'];
  }
  if (!textDetectionOptions.minConfidence) {
    textDetectionOptions.minConfidence = config['MIN_CONFIDENCE'];
  }
  if (!textDetectionOptions.resizeLength) {
    textDetectionOptions.resizeLength = config['RESIZE_LENGTH'];
  }
  if (!textDetectionOptions.processPoints) {
    textDetectionOptions.processPoints = minAreaRect;
  }
  const {minTextBoxArea, minConfidence, resizeLength, processPoints} =
      textDetectionOptions;

  const [heightScalingFactor, widthScalingFactor] =
      computeScalingFactors(originalHeight, originalWidth, resizeLength);

  const [height, width, numOfKernels] = kernelScores.shape;
  const kernels = tf.tidy(() => {
    const one = tf.ones([height, width, numOfKernels], 'int32');
    const zero = tf.zeros([height, width, numOfKernels], 'int32');
    const threshold = tf.scalar(minConfidence, 'float32');
    return tf.where(kernelScores.greater(threshold), one, zero) as tf.Tensor3D;
  });
  const {segmentationMapBuffer, recognizedLabels} =
      await progressiveScaleExpansion(
          kernels, minTextBoxArea * heightScalingFactor * widthScalingFactor);
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
      await resizedSegmentationMap.array() as number[][];

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
  const clip = (size: number, edge: number) => (size > edge ? edge : size);
  Object.keys(points).forEach((labelStr) => {
    const label = Number(labelStr);
    const box = processPoints(points[label]);
    // const box = points[label];
    for (let pointIdx = 0; pointIdx < box.length; ++pointIdx) {
      const point = box[pointIdx];
      const scaledX = clip(point.x / widthScalingFactor, originalWidth);
      const scaledY = clip(point.y / heightScalingFactor, originalHeight);
      box[pointIdx] = new Point(scaledX, scaledY);
    }
    boxes.push(box as Box);
  });
  return boxes;
};

export const computeScalingFactors =
    (height: number, width: number, resizeLength: number): [number, number] => {
      const maxSide = Math.max(width, height);
      const ratio = maxSide > resizeLength ? resizeLength / maxSide : 1;

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

export const resize = (input: TextDetectionInput,
                       resizeLength?: number): tf.Tensor3D => {
  return tf.tidy(() => {
    if (!resizeLength) {
      resizeLength = config['RESIZE_LENGTH'];
    }
    const image: tf.Tensor3D = (input instanceof tf.Tensor ?
                                    input :
                                    tf.browser.fromPixels(
                                        input as ImageData | HTMLImageElement |
                                        HTMLCanvasElement | HTMLVideoElement))
                                   .toFloat();

    const [height, width] = image.shape;
    const [heightScalingFactor, widthScalingFactor] =
        computeScalingFactors(height, width, resizeLength);
    const targetHeight = Math.round(height * heightScalingFactor);
    const targetWidth = Math.round(width * widthScalingFactor);
    const processedImage =
        tf.image.resizeBilinear(image, [targetHeight, targetWidth]);

    return processedImage;
  });
};
