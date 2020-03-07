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

import {connectedPartIndices} from './keypoints';
import {InputResolution, Keypoint, Padding, Pose, PosenetInput, PoseNetInternalResolution, PoseNetOutputStride, TensorBuffer3D, Vector2D} from './types';

function eitherPointDoesntMeetConfidence(
    a: number, b: number, minConfidence: number): boolean {
  return (a < minConfidence || b < minConfidence);
}

export function getAdjacentKeyPoints(
    keypoints: Keypoint[], minConfidence: number): Keypoint[][] {
  return connectedPartIndices.reduce(
      (result: Keypoint[][], [leftJoint, rightJoint]): Keypoint[][] => {
        if (eitherPointDoesntMeetConfidence(
                keypoints[leftJoint].score, keypoints[rightJoint].score,
                minConfidence)) {
          return result;
        }

        result.push([keypoints[leftJoint], keypoints[rightJoint]]);

        return result;
      }, []);
}

const {NEGATIVE_INFINITY, POSITIVE_INFINITY} = Number;
export function getBoundingBox(keypoints: Keypoint[]):
    {maxX: number, maxY: number, minX: number, minY: number} {
  return keypoints.reduce(({maxX, maxY, minX, minY}, {position: {x, y}}) => {
    return {
      maxX: Math.max(maxX, x),
      maxY: Math.max(maxY, y),
      minX: Math.min(minX, x),
      minY: Math.min(minY, y)
    };
  }, {
    maxX: NEGATIVE_INFINITY,
    maxY: NEGATIVE_INFINITY,
    minX: POSITIVE_INFINITY,
    minY: POSITIVE_INFINITY
  });
}

export function getBoundingBoxPoints(keypoints: Keypoint[]): Vector2D[] {
  const {minX, minY, maxX, maxY} = getBoundingBox(keypoints);
  return [
    {x: minX, y: minY}, {x: maxX, y: minY}, {x: maxX, y: maxY},
    {x: minX, y: maxY}
  ];
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

export function toValidInputResolution(
    inputResolution: number, outputStride: PoseNetOutputStride): number {
  if (isValidInputResolution(inputResolution, outputStride)) {
    return inputResolution;
  }

  return Math.floor(inputResolution / outputStride) * outputStride + 1;
}

function validateInputResolution(inputResolution: InputResolution) {
  tf.util.assert(
      typeof inputResolution === 'number' ||
          typeof inputResolution === 'object',
      () => `Invalid inputResolution ${inputResolution}. ` +
          `Should be a number or an object with width and height`);

  if (typeof inputResolution === 'object') {
    tf.util.assert(
        typeof inputResolution.width === 'number',
        () => `inputResolution.width has a value of ${
            inputResolution.width} which is invalid; it must be a number`);
    tf.util.assert(
        typeof inputResolution.height === 'number',
        () => `inputResolution.height has a value of ${
            inputResolution.height} which is invalid; it must be a number`);
  }
}

export function getValidInputResolutionDimensions(
    inputResolution: InputResolution,
    outputStride: PoseNetOutputStride): [number, number] {
  validateInputResolution(inputResolution);
  if (typeof inputResolution === 'object') {
    return [
      toValidInputResolution(inputResolution.height, outputStride),
      toValidInputResolution(inputResolution.width, outputStride),
    ];
  } else {
    return [
      toValidInputResolution(inputResolution, outputStride),
      toValidInputResolution(inputResolution, outputStride),
    ];
  }
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

const MIN_RESOLUTION_PERCENTAGE = 0.1;
const MAX_RESOLUTION_PERCENTAGE = 4.0;

function toInternalResolutionPercentage(
    internalResolution: PoseNetInternalResolution): number {
  if (internalResolution == null) {
    return INTERNAL_RESOLUTION_PERCENTAGES[INTERNAL_RESOLUTION_STRING_OPTIONS
                                               .medium];
  }
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
            internalResolution <= MAX_RESOLUTION_PERCENTAGE &&
            internalResolution >= MIN_RESOLUTION_PERCENTAGE,
        () => `inputResolution must be a string or number between 0 and ${
            MAX_RESOLUTION_PERCENTAGE}, but was ${internalResolution}`);

    return internalResolution;
  }
}

export function toInputResolutionHeightAndWidth(
    inputResolutionFromModelConfig: [number, number],
    internalResolution: PoseNetInternalResolution,
    outputStride: PoseNetOutputStride,
    [inputHeight, inputWidth]: [number, number]): [number, number] {
  if (inputResolutionFromModelConfig) {
    return inputResolutionFromModelConfig;
  }

  const internalResolutionPercentage =
      toInternalResolutionPercentage(internalResolution);

  return [
    toValidInputResolution(
        inputHeight * internalResolutionPercentage, outputStride),
    toValidInputResolution(
        inputWidth * internalResolutionPercentage, outputStride)
  ];
}

const VALID_OUTPUT_STRIDES: PoseNetOutputStride[] = [8, 16, 32];
export function assertValidOutputStride(outputStride: PoseNetOutputStride) {
  tf.util.assert(
      typeof outputStride === 'number', () => 'outputStride is not a number');
  tf.util.assert(
      VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0,
      () => `outputStride of ${outputStride} is invalid. ` +
          `It must be either 8, 16, or 32`);
}

function isValidInputResolution(
    resolution: number, outputStride: number): boolean {
  return (resolution - 1) % outputStride === 0;
}

export function getInputTensorDimensions(input: PosenetInput):
    [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

export function toInputTensor(input: PosenetInput) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

export function toResizedInputTensor(
    input: PosenetInput, resizeHeight: number, resizeWidth: number,
    flipHorizontal: boolean): tf.Tensor3D {
  return tf.tidy(() => {
    const imageTensor = toInputTensor(input);

    if (flipHorizontal) {
      return imageTensor.reverse(1).resizeBilinear([resizeHeight, resizeWidth]);
    } else {
      return imageTensor.resizeBilinear([resizeHeight, resizeWidth]);
    }
  });
}

export function padAndResizeTo(
    input: PosenetInput, [targetH, targetW]: [number, number]):
    {resized: tf.Tensor3D, padding: Padding} {
  const [height, width] = getInputTensorDimensions(input);
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
