/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 *
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

import {BodyPixInput, BodyPixOutputStride, Padding} from './types';
import {Pose, TensorBuffer3D} from './types';
import {BodyPixInternalResolution} from './types';

function getSizeFromImageLikeElement(input: HTMLImageElement|
                                     HTMLCanvasElement): [number, number] {
  if (input.offsetHeight !== 0 && input.offsetWidth !== 0) {
    return [input.offsetHeight, input.offsetWidth];
  } else if (input.height != null && input.width != null) {
    return [input.height, input.width];
  } else {
    throw new Error(
        `HTMLImageElement must have height and width attributes set.`);
  }
}

function getSizeFromVideoElement(input: HTMLVideoElement): [number, number] {
  if (input.hasAttribute('height') && input.hasAttribute('width')) {
    // Prioritizes user specified height and width.
    // We can't test the .height and .width properties directly,
    // because they evaluate to 0 if unset.
    return [input.height, input.width];
  } else {
    return [input.videoHeight, input.videoWidth];
  }
}

export function getInputSize(input: BodyPixInput): [number, number] {
  if ((typeof (HTMLCanvasElement) !== 'undefined' &&
       input instanceof HTMLCanvasElement) ||
      (typeof (HTMLImageElement) !== 'undefined' &&
       input instanceof HTMLImageElement)) {
    return getSizeFromImageLikeElement(input);
  } else if (typeof (ImageData) !== 'undefined' && input instanceof ImageData) {
    return [input.height, input.width];
  } else if (
      typeof (HTMLVideoElement) !== 'undefined' &&
      input instanceof HTMLVideoElement) {
    return getSizeFromVideoElement(input);
  } else if (input instanceof tf.Tensor) {
    return [input.shape[0], input.shape[1]];
  } else {
    throw new Error(`error: Unknown input type: ${input}.`);
  }
}

function isValidInputResolution(
    resolution: number, outputStride: number): boolean {
  return (resolution - 1) % outputStride === 0;
}

export function toValidInputResolution(
    inputResolution: number, outputStride: BodyPixOutputStride): number {
  if (isValidInputResolution(inputResolution, outputStride)) {
    return inputResolution;
  }

  return Math.floor(inputResolution / outputStride) * outputStride + 1;
}

const INTERNAL_RESOLUTION_STRING_OPTIONS = {
  low: 'low',
  medium: 'medium',
  high: 'high',
  full: 'full'
};

const INTERNAL_RESOLUTION_PERCENTAGES = {
  [INTERNAL_RESOLUTION_STRING_OPTIONS.low]: 0.25,
  [INTERNAL_RESOLUTION_STRING_OPTIONS.medium]: 0.5,
  [INTERNAL_RESOLUTION_STRING_OPTIONS.high]: 0.75,
  [INTERNAL_RESOLUTION_STRING_OPTIONS.full]: 1.0
};

const MIN_INTERNAL_RESOLUTION = 0.1;
const MAX_INTERNAL_RESOLUTION = 2.0;

function toInternalResolutionPercentage(
    internalResolution: BodyPixInternalResolution): number {
  if (typeof internalResolution === 'string') {
    const result = INTERNAL_RESOLUTION_PERCENTAGES[internalResolution];

    tf.util.assert(
        typeof result === 'number',
        () => `string value of inputResolution must be one of ${
            Object.values(INTERNAL_RESOLUTION_STRING_OPTIONS)
                .join(',')} but was ${internalResolution}.`);
    return result;
  } else {
    tf.util.assert(
        typeof internalResolution === 'number' &&
            internalResolution <= MAX_INTERNAL_RESOLUTION &&
            internalResolution >= MIN_INTERNAL_RESOLUTION,
        () =>
            `inputResolution must be a string or number between ${
                MIN_INTERNAL_RESOLUTION} and ${MAX_INTERNAL_RESOLUTION}, but ` +
            `was ${internalResolution}`);

    return internalResolution;
  }
}

export function toInputResolutionHeightAndWidth(
    internalResolution: BodyPixInternalResolution,
    outputStride: BodyPixOutputStride,
    [inputHeight, inputWidth]: [number, number]): [number, number] {
  const internalResolutionPercentage =
      toInternalResolutionPercentage(internalResolution);

  return [
    toValidInputResolution(
        inputHeight * internalResolutionPercentage, outputStride),
    toValidInputResolution(
        inputWidth * internalResolutionPercentage, outputStride)
  ];
}

export function toInputTensor(input: BodyPixInput) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

export function resizeAndPadTo(
    imageTensor: tf.Tensor3D, [targetH, targetW]: [number, number],
    flipHorizontal = false): {
  resizedAndPadded: tf.Tensor3D,
  paddedBy: [[number, number], [number, number]]
} {
  const [height, width] = imageTensor.shape;

  const targetAspect = targetW / targetH;
  const aspect = width / height;

  let resizeW: number;
  let resizeH: number;
  let padL: number;
  let padR: number;
  let padT: number;
  let padB: number;

  if (aspect > targetAspect) {
    // resize to have the larger dimension match the shape.
    resizeW = targetW;
    resizeH = Math.ceil(resizeW / aspect);

    const padHeight = targetH - resizeH;
    padL = 0;
    padR = 0;
    padT = Math.floor(padHeight / 2);
    padB = targetH - (resizeH + padT);
  } else {
    resizeH = targetH;
    resizeW = Math.ceil(targetH * aspect);

    const padWidth = targetW - resizeW;
    padL = Math.floor(padWidth / 2);
    padR = targetW - (resizeW + padL);
    padT = 0;
    padB = 0;
  }

  const resizedAndPadded = tf.tidy(() => {
    // resize to have largest dimension match image
    let resized: tf.Tensor3D;
    if (flipHorizontal) {
      resized = imageTensor.reverse(1).resizeBilinear([resizeH, resizeW]);
    } else {
      resized = imageTensor.resizeBilinear([resizeH, resizeW]);
    }

    const padded = tf.pad3d(resized, [[padT, padB], [padL, padR], [0, 0]]);

    return padded;
  });

  return {resizedAndPadded, paddedBy: [[padT, padB], [padL, padR]]};
}

export function scaleAndCropToInputTensorShape(
    tensor: tf.Tensor3D,
    [inputTensorHeight, inputTensorWidth]: [number, number],
    [resizedAndPaddedHeight, resizedAndPaddedWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    applySigmoidActivation = false): tf.Tensor3D {
  return tf.tidy(() => {
    let inResizedAndPadded: tf.Tensor3D = tensor.resizeBilinear(
        [resizedAndPaddedHeight, resizedAndPaddedWidth], true);

    if (applySigmoidActivation) {
      inResizedAndPadded = inResizedAndPadded.sigmoid();
    }

    return removePaddingAndResizeBack(
        inResizedAndPadded, [inputTensorHeight, inputTensorWidth],
        [[padT, padB], [padL, padR]]);
  });
}

export function removePaddingAndResizeBack(
    resizedAndPadded: tf.Tensor3D,
    [originalHeight, originalWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    tf.Tensor3D {
  return tf.tidy(() => {
    const batchedImage: tf.Tensor4D = resizedAndPadded.expandDims();
    return tf.image
        .cropAndResize(
            batchedImage, [[
              padT / (originalHeight + padT + padB - 1.0),
              padL / (originalWidth + padL + padR - 1.0),
              (padT + originalHeight - 1.0) /
                  (originalHeight + padT + padB - 1.0),
              (padL + originalWidth - 1.0) / (originalWidth + padL + padR - 1.0)
            ]],
            [0], [originalHeight, originalWidth])
        .squeeze([0]);
  });
}

export function resize2d(
    tensor: tf.Tensor2D, resolution: [number, number],
    nearestNeighbor?: boolean): tf.Tensor2D {
  return tf.tidy(() => {
    const batchedImage: tf.Tensor4D = tensor.expandDims(2);
    return batchedImage.resizeBilinear(resolution, nearestNeighbor).squeeze();
  });
}

export function padAndResizeTo(
    input: BodyPixInput, [targetH, targetW]: [number, number]):
    {resized: tf.Tensor3D, padding: Padding} {
  const [height, width] = getInputSize(input);
  const targetAspect = targetW / targetH;
  const aspect = width / height;
  let [padT, padB, padL, padR] = [0, 0, 0, 0];
  if (aspect < targetAspect) {
    // pads the width
    padT = 0;
    padB = 0;
    padL = Math.round(0.5 * (targetAspect * height - width));
    padR = Math.round(0.5 * (targetAspect * height - width));
  } else {
    // pads the height
    padT = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padB = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padL = 0;
    padR = 0;
  }

  const resized: tf.Tensor3D = tf.tidy(() => {
    let imageTensor = toInputTensor(input);
    imageTensor = tf.pad3d(imageTensor, [[padT, padB], [padL, padR], [0, 0]]);

    return imageTensor.resizeBilinear([targetH, targetW]);
  });

  return {resized, padding: {top: padT, left: padL, right: padR, bottom: padB}};
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => tensor.buffer()));
}

export function scalePose(
    pose: Pose, scaleY: number, scaleX: number, offsetY = 0,
    offsetX = 0): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(({score, part, position}) => ({
                                    score,
                                    part,
                                    position: {
                                      x: position.x * scaleX + offsetX,
                                      y: position.y * scaleY + offsetY
                                    }
                                  }))
  };
}

export function scalePoses(
    poses: Pose[], scaleY: number, scaleX: number, offsetY = 0, offsetX = 0) {
  if (scaleX === 1 && scaleY === 1 && offsetY === 0 && offsetX === 0) {
    return poses;
  }
  return poses.map(pose => scalePose(pose, scaleY, scaleX, offsetY, offsetX));
}

export function flipPoseHorizontal(pose: Pose, imageWidth: number): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
        ({score, part, position}) => ({
          score,
          part,
          position: {x: imageWidth - 1 - position.x, y: position.y}
        }))
  };
}

export function flipPosesHorizontal(poses: Pose[], imageWidth: number) {
  if (imageWidth <= 0) {
    return poses;
  }
  return poses.map(pose => flipPoseHorizontal(pose, imageWidth));
}

export function scaleAndFlipPoses(
    poses: Pose[], [height, width]: [number, number],
    [inputResolutionHeight, inputResolutionWidth]: [number, number],
    padding: Padding, flipHorizontal: boolean): Pose[] {
  const scaleY =
      (height + padding.top + padding.bottom) / (inputResolutionHeight);
  const scaleX =
      (width + padding.left + padding.right) / (inputResolutionWidth);

  const scaledPoses =
      scalePoses(poses, scaleY, scaleX, -padding.top, -padding.left);

  if (flipHorizontal) {
    return flipPosesHorizontal(scaledPoses, width);
  } else {
    return scaledPoses;
  }
}
