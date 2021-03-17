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
import {InputResolution, PoseDetectorInput} from '../types';
import {ImageSize, Padding, ValueTransform} from './interfaces/common_interfaces';
import {Rect} from './interfaces/shape_interfaces';

export function getImageSize(input: PoseDetectorInput): ImageSize {
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
export function toImageTensor(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                              HTMLImageElement|HTMLCanvasElement) {
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
 * @param subRect The rectangle to generate the projective transformation matrix
 *     for.
 * @param imageSize The original image height and width.
 * @param flipHorizontally Whether flip the image horizontally.
 * @param inputResolution The target height and width.
 */
export function getProjectiveTransformMatrix(
    subRect: Rect, imageSize: ImageSize, flipHorizontally: boolean,
    inputResolution: InputResolution):
    [number, number, number, number, number, number, number, number] {
  validateSize(inputResolution, 'inputResolution');

  // Ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tensor/image_to_tensor_utils.cc
  // The resulting matrix is multiplication of below matrices:
  // M = postScaleMatrix * translateMatrix * rotateMatrix * flipMatrix *
  //     scaleMatrix * initialTranslateMatrix
  //
  // For any point in the transformed image p, we can use the above matrix to
  // calculate the projected point in the original image p'. So that:
  // p' = p * M;
  // Note: The transform matrix below assumes image coordinates is normalized
  // to [0, 1] range.

  // postScaleMatrix: Matrix to scale x, y to [0, 1] range
  //   | g  0  0 |
  //   | 0  h  0 |
  //   | 0  0  1 |
  const g = 1 / imageSize.width;
  const h = 1 / imageSize.height;

  // translateMatrix: Matrix to move the center to the subRect center.
  //   | 1  0  e |
  //   | 0  1  f |
  //   | 0  0  1 |
  const e = subRect.xCenter;
  const f = subRect.yCenter;

  // rotateMatrix: Matrix to do rotate the image around the subRect center.
  //   | c -d  0 |
  //   | d  c  0 |
  //   | 0  0  1 |
  const c = Math.cos(subRect.rotation);
  const d = Math.sin(subRect.rotation);

  // flipMatrix: Matrix for optional horizontal flip around the subRect center.
  //   | fl 0  0 |
  //   | 0  1  0 |
  //   | 0  0  1 |
  const flip = flipHorizontally ? -1 : 1;

  // scaleMatrix: Matrix to scale x, y to subRect size.
  //   | a  0  0 |
  //   | 0  b  0 |
  //   | 0  0  1 |
  const a = subRect.width;
  const b = subRect.height;

  // initialTranslateMatrix: Matrix convert x, y to [-0.5, 0.5] range.
  //   | 1  0 -0.5 |
  //   | 0  1 -0.5 |
  //   | 0  0  1   |

  // M is a 3 by 3 matrix denoted by:
  // | a0  a1  a2 |
  // | b0  b1  b2 |
  // | 0   0   1  |
  // To use M with regular x, y coordinates, we need to normalize them first.
  // Because x' = a0 * x + a1 * y + a2, y' = b0 * x + b1 * y + b2,
  // we need to use factor (1/inputResolution.width) to normalize x for a0 and
  // b0, similarly we need to use factor (1/inputResolution.height) to normalize
  // y for a1 and b1.
  // Also at the end, we need to de-normalize x' and y' to regular coordinates.
  // So we need to use factor imageSize.width for a0, a1 and a2, similarly
  // we need to use factor imageSize.height for b0, b1 and b2.
  const a0 = (1 / inputResolution.width) * a * c * flip * g * imageSize.width;
  const a1 = (1 / inputResolution.height) * -b * d * g * imageSize.width;
  const a2 = (-0.5 * a * c * flip + 0.5 * b * d + e) * g * imageSize.width;
  const b0 = (1 / inputResolution.width) * a * d * flip * h * imageSize.height;
  const b1 = (1 / inputResolution.height) * b * c * h * imageSize.height;
  const b2 = (-0.5 * b * c - 0.5 * a * d * flip + f) * h * imageSize.height;

  return [a0, a1, a2, b0, b1, b2, 0, 0];
}

function validateSize(size: {width: number, height: number}, name: string) {
  tf.util.assert(size.width !== 0, () => `${name} width cannot be 0.`);
  tf.util.assert(size.height !== 0, () => `${name} height cannot be 0.`);
}
