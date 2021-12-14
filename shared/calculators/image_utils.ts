/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import {Matrix4x4} from './calculate_inverse_matrix';

import {ImageSize, InputResolution, Padding, PixelInput, ValueTransform} from './interfaces/common_interfaces';
import {Rect} from './interfaces/shape_interfaces';

export function getImageSize(input: PixelInput): ImageSize {
  if (input instanceof tf.Tensor) {
    return {height: input.shape[0], width: input.shape[1]};
  } else {
    return {height: input.height, width: input.width};
  }
}

/**
 * Normalizes the provided angle to the range -pi to pi.
 * @param angle The angle in radians to be normalized.
 */
export function normalizeRadians(angle: number): number {
  return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
}

/**
 * Transform value ranges.
 * @param fromMin Min of original value range.
 * @param fromMax Max of original value range.
 * @param toMin New min of transformed value range.
 * @param toMax New max of transformed value range.
 */
export function transformValueRange(
    fromMin: number, fromMax: number, toMin: number,
    toMax: number): ValueTransform {
  const fromRange = fromMax - fromMin;
  const toRange = toMax - toMin;

  if (fromRange === 0) {
    throw new Error(
        `Original min and max are both ${fromMin}, range cannot be 0.`);
  }

  const scale = toRange / fromRange;
  const offset = toMin - fromMin * scale;
  return {scale, offset};
}

/**
 * Convert an image to an image tensor representation.
 *
 * The image tensor has a shape [1, height, width, colorChannel].
 *
 * @param input An image, video frame, or image tensor.
 */
export function toImageTensor(input: PixelInput) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

/**
 * Padding ratio of left, top, right, bottom, based on the output dimensions.
 *
 * The padding values are non-zero only when the "keep_aspect_ratio" is true.
 *
 * For instance, when the input image is 10x10 (width x height) and the
 * output dimensions is 20x40 and "keep_aspect_ratio" is true, we should scale
 * the input image to 20x20 and places it in the middle of the output image with
 * an equal padding of 10 pixels at the top and the bottom. The result is
 * therefore {left: 0, top: 0.25, right: 0, bottom: 0.25} (10/40 = 0.25f).
 * @param roi The original rectangle to pad.
 * @param targetSize The target width and height of the result rectangle.
 * @param keepAspectRatio Whether keep aspect ratio. Default to false.
 */
export function padRoi(
    roi: Rect, targetSize: InputResolution, keepAspectRatio = false): Padding {
  if (!keepAspectRatio) {
    return {top: 0, left: 0, right: 0, bottom: 0};
  }

  const targetH = targetSize.height;
  const targetW = targetSize.width;

  validateSize(targetSize, 'targetSize');
  validateSize(roi, 'roi');

  const tensorAspectRatio = targetH / targetW;
  const roiAspectRatio = roi.height / roi.width;
  let newWidth;
  let newHeight;
  let horizontalPadding = 0;
  let verticalPadding = 0;
  if (tensorAspectRatio > roiAspectRatio) {
    // pad height;
    newWidth = roi.width;
    newHeight = roi.width * tensorAspectRatio;
    verticalPadding = (1 - roiAspectRatio / tensorAspectRatio) / 2;
  } else {
    // pad width.
    newWidth = roi.height / tensorAspectRatio;
    newHeight = roi.height;
    horizontalPadding = (1 - tensorAspectRatio / roiAspectRatio) / 2;
  }

  roi.width = newWidth;
  roi.height = newHeight;

  return {
    top: verticalPadding,
    left: horizontalPadding,
    right: horizontalPadding,
    bottom: verticalPadding
  };
}

/**
 * Get the rectangle information of an image, including xCenter, yCenter, width,
 * height and rotation.
 *
 * @param imageSize imageSize is used to calculate the rectangle.
 * @param normRect Optional. If normRect is not null, it will be used to get
 *     a subarea rectangle information in the image. `imageSize` is used to
 *     calculate the actual non-normalized coordinates.
 */
export function getRoi(imageSize: ImageSize, normRect?: Rect): Rect {
  if (normRect) {
    return {
      xCenter: normRect.xCenter * imageSize.width,
      yCenter: normRect.yCenter * imageSize.height,
      width: normRect.width * imageSize.width,
      height: normRect.height * imageSize.height,
      rotation: normRect.rotation
    };
  } else {
    return {
      xCenter: 0.5 * imageSize.width,
      yCenter: 0.5 * imageSize.height,
      width: imageSize.width,
      height: imageSize.height,
      rotation: 0
    };
  }
}

/**
 * Generate the projective transformation matrix to be used for `tf.transform`.
 *
 * See more documentation in `tf.transform`.
 *
 * @param matrix The transformation matrix mapping subRect to rect, can be
 *     computed using `getRotatedSubRectToRectTransformMatrix` calculator.
 * @param imageSize The original image height and width.
 * @param inputResolution The target height and width.
 */
export function getProjectiveTransformMatrix(
    matrix: Matrix4x4, imageSize: ImageSize, inputResolution: InputResolution):
    [number, number, number, number, number, number, number, number] {
  validateSize(inputResolution, 'inputResolution');

  // To use M with regular x, y coordinates, we need to normalize them first.
  // Because x' = a0 * x + a1 * y + a2, y' = b0 * x + b1 * y + b2,
  // we need to use factor (1/inputResolution.width) to normalize x for a0 and
  // b0, similarly we need to use factor (1/inputResolution.height) to normalize
  // y for a1 and b1.
  // Also at the end, we need to de-normalize x' and y' to regular coordinates.
  // So we need to use factor imageSize.width for a0, a1 and a2, similarly
  // we need to use factor imageSize.height for b0, b1 and b2.
  const a0 = (1 / inputResolution.width) * matrix[0][0] * imageSize.width;
  const a1 = (1 / inputResolution.height) * matrix[0][1] * imageSize.width;
  const a2 = matrix[0][3] * imageSize.width;
  const b0 = (1 / inputResolution.width) * matrix[1][0] * imageSize.height;
  const b1 = (1 / inputResolution.height) * matrix[1][1] * imageSize.height;
  const b2 = matrix[1][3] * imageSize.height;

  return [a0, a1, a2, b0, b1, b2, 0, 0];
}

function validateSize(size: {width: number, height: number}, name: string) {
  tf.util.assert(size.width !== 0, () => `${name} width cannot be 0.`);
  tf.util.assert(size.height !== 0, () => `${name} height cannot be 0.`);
}
