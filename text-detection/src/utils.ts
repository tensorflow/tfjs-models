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

export function getURL(quantizationBytes: QuantizationBytes) {
  // #TODO(tfjs): Change the versioning convention
  return `${config['BASE_PATH']}/${
      ([1, 2].indexOf(quantizationBytes) !== -1) ?
          `quantized/${quantizationBytes}/` :
          ''}${config['MODEL_VERSION']}/model.json`;
};

export async function convertKernelsToBoxes(
    kernelLogits: tf.Tensor3D,
    originalHeight: number,
    originalWidth: number,
    textDetectionOptions: TextDetectionOptions = {
      debug: config['DEBUG'],
      minPixelSalience: config['MIN_PIXEL_SALIENCE'],
      minTextBoxArea: config['MIN_TEXTBOX_AREA'],
      minTextConfidence: config['MIN_TEXT_CONFIDENCE'],
      processPoints: minAreaRect,
      resizeLength: config['RESIZE_LENGTH'],
    },
    ): Promise<Box[]> {
  textDetectionOptions = {
    debug: config['DEBUG'],
    minPixelSalience: config['MIN_PIXEL_SALIENCE'],
    minTextBoxArea: config['MIN_TEXTBOX_AREA'],
    minTextConfidence: config['MIN_TEXT_CONFIDENCE'],
    processPoints: minAreaRect,
    resizeLength: config['RESIZE_LENGTH'],
    ...textDetectionOptions
  };

  const {
    minTextBoxArea,
    minPixelSalience,
    minTextConfidence,
    resizeLength,
    processPoints,
    debug
  } = textDetectionOptions;

  const [kernelHeight, kernelWidth] = kernelLogits.shape;
  const score = tf.tidy(() => {
    const text = kernelLogits.stridedSlice(
        [0, 0, 0], [kernelHeight, kernelWidth, 1], [1, 1, 1]);
    return text.sigmoid();
  });

  if (debug) {
    console.log('The text scores are:');
    score.print(true);
  }

  const kernels = tf.tidy(() => {
    const labels =
        (tf.sign(tf.sub(kernelLogits, minPixelSalience)).add(1.0)).div(2.0);
    const text = labels.stridedSlice(
        [0, 0, 0], [kernelHeight, kernelWidth, 1], [1, 1, 1]);
    if (debug) {
      const nonzero = tf.tidy(() => {
        return text.greater(0).sum();
      });
      console.log(`The text mask ${nonzero} non-zero pixels.`);
    }
    return labels.mul(text) as tf.Tensor3D;
  });

  const [heightScalingFactor, widthScalingFactor] =
      computeScalingFactors(originalHeight, originalWidth, resizeLength);

  const {segmentationMapBuffer, recognizedLabels} =
      await progressiveScaleExpansion(kernels, minTextBoxArea);
  tf.dispose(kernels);

  if (debug) {
    console.log('The recognized labels are:', Array.from(recognizedLabels));
  }

  if (recognizedLabels.size === 0) {
    tf.dispose(score);
    return [];
  }

  const scoreData = await score.array() as number[][];
  tf.dispose(score);

  const labelScores: {[label: number]: [number, number]} = {};
  for (let rowIdx = 0; rowIdx < kernelHeight; ++rowIdx) {
    for (let colIdx = 0; colIdx < kernelWidth; ++colIdx) {
      const label = segmentationMapBuffer.get(rowIdx, colIdx);
      const score = Number(scoreData[rowIdx][colIdx]);
      if (recognizedLabels.has(label)) {
        if (!labelScores.hasOwnProperty(label)) {
          labelScores[label] = [score, 1];
        } else {
          const [oldScore, oldCount] = labelScores[label];
          labelScores[label] = [oldScore + score, oldCount + 1];
        }
      }
    }
  }

  if (debug) {
    console.log('The label scores are:');
    Object.keys(labelScores).forEach((label: string) => {
      const [score, count] = labelScores[Number(label)];
      console.log(`\t${label}: score ${score}, count ${count}`);
    });
  }

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

  const points: {[label: number]: Point[]} = {};
  for (let rowIdx = 0; rowIdx < targetHeight; ++rowIdx) {
    for (let columnIdx = 0; columnIdx < targetWidth; ++columnIdx) {
      const label = resizedSegmentationMapData[rowIdx][columnIdx];
      if (recognizedLabels.has(label)) {
        const [totalScore, totalCount] = labelScores[label];
        const labelScore = totalScore / totalCount;
        if (labelScore > minTextConfidence) {
          if (!points[label]) {
            points[label] = [];
          }
          points[label].push(new Point(columnIdx, rowIdx));
        }
      }
    }
  }

  const boxes: Box[] = [];

  const clip = (size: number, edge: number) => (size > edge ? edge : size);
  Object.keys(points).forEach((labelStr) => {
    const label = Number(labelStr);
    const box = processPoints(points[label]);
    for (let pointIdx = 0; pointIdx < box.length; ++pointIdx) {
      const point = box[pointIdx];
      const scaledX = clip(point.x / widthScalingFactor, originalWidth);
      const scaledY = clip(point.y / heightScalingFactor, originalHeight);
      box[pointIdx] = new Point(scaledX, scaledY);
    }
    boxes.push(box as Box);
  });
  return boxes;
}

export function computeScalingFactors(
    height: number, width: number, resizeLength: number): [number, number] {
  const maxSide = Math.max(width, height);
  const ratio = maxSide > resizeLength ? resizeLength / maxSide : 1;

  const getScalingFactor = (side: number) => {
    const roundedSide = Math.round(side * ratio);
    return (roundedSide % 32 === 0 ?
                roundedSide :
                Math.max(Math.floor(roundedSide / 32) * 32, 32)) /
        side;
  };

  const heightScalingRatio = getScalingFactor(height);
  const widthScalingRatio = getScalingFactor(width);
  return [heightScalingRatio, widthScalingRatio];
}

export function resize(
    input: TextDetectionInput, resizeLength?: number): tf.Tensor3D {
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
}
