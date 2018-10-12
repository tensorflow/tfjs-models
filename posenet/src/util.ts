/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';

import {connectedPartIndices} from './keypoints';
import {OutputStride} from './mobilenet';
import {Keypoint, Pose, PosenetInput, TensorBuffer3D, Vector2D} from './types';

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

export async function toTensorBuffer<rank extends tf.Rank>(
    tensor: tf.Tensor<rank>,
    type: 'float32'|'int32' = 'float32'): Promise<tf.TensorBuffer<rank>> {
  const tensorData = await tensor.data();

  return new tf.TensorBuffer<rank>(tensor.shape, type, tensorData);
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
}

export function scalePose(pose: Pose, scaleY: number, scaleX: number): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
        ({score, part, position}) => ({
          score,
          part,
          position: {x: position.x * scaleX, y: position.y * scaleY}
        }))
  };
}

export function translateAndScalePose(
    pose: Pose,
    translateY: number,
    translateX: number,
    scaleY: number,
    scaleX: number,
) {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(({score, part, position}) => ({
                                    score,
                                    part,
                                    position: {
                                      x: (position.x + translateX) * scaleX,
                                      y: (position.y + translateY) * scaleY
                                    }
                                  }))
  };
}

export function scalePoses(poses: Pose[], scaleY: number, scaleX: number) {
  return poses.map(pose => scalePose(pose, scaleY, scaleX));
}

export function getValidResolution(
    imageScaleFactor: number, inputDimension: number,
    outputStride: OutputStride): number {
  const evenResolution = inputDimension * imageScaleFactor - 1;

  return evenResolution - (evenResolution % outputStride) + 1;
}
export function resize2d(
    tensor: tf.Tensor2D, resolution: [number, number],
    nearestNeighbor?: boolean): tf.Tensor2D {
  return tf.tidy(
      () => (tensor.expandDims(2) as tf.Tensor3D)
                .resizeBilinear(resolution, nearestNeighbor)
                .squeeze() as tf.Tensor2D);
}

export function getInputTensorDimensions(input: PosenetInput):
    [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

export function toInputTensor(input: PosenetInput) {
  return input instanceof tf.Tensor ? input : tf.fromPixels(input);
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

export function cropAndResizeTo(
    input: PosenetInput, [targetHeight, targetWidth]: number[]) {
  const [height, width] = getInputTensorDimensions(input);
  const imageTensor = toInputTensor(input);

  const targetAspect = targetWidth / targetHeight;
  const aspect = width / height;

  let croppedW: number;
  let croppedH: number;

  if (aspect > targetAspect) {
    // crop width to get aspect
    croppedW = Math.round(height * targetAspect);
    croppedH = height;
  } else {
    croppedH = Math.round(width / targetAspect);
    croppedW = width;
  }

  const startCropTop = Math.floor((height - croppedH) / 2);
  const startCropLeft = Math.floor((width - croppedW) / 2);

  const resizedWidth = targetWidth;
  const resizedHeight = targetHeight;

  const croppedAndResized = tf.tidy(() => {
    const cropped = tf.slice3d(
        imageTensor, [startCropTop, startCropLeft, 0], [croppedH, croppedW, 3]);

    return cropped.resizeBilinear([resizedHeight, resizedWidth]);
  });

  return {
    croppedAndResized,
    resizedDimensions: [resizedHeight, resizedWidth],
    crop: [startCropTop, startCropLeft, croppedH, croppedW]
  };
}

export function resizeAndPadTo(
    input: PosenetInput, [targetH, targetW]: [number, number],
    flipHorizontal = false): {
  resizedAndPadded: tf.Tensor3D,
  paddedBy: [[number, number], [number, number]]
} {
  const [height, width] = getInputTensorDimensions(input);
  const imageTensor = toInputTensor(input);

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
    resizeW = Math.ceil(targetH / aspect);

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
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    tf.Tensor3D {
  return tf.tidy(() => {
    const inResizedAndPaddedSize = tensor.resizeBilinear(
        [resizedAndPaddedHeight, resizedAndPaddedWidth], true);

    return removePaddingAndResizeBack(
        inResizedAndPaddedSize, [inputTensorHeight, inputTensorWidth],
        [[padT, padB], [padL, padR]]);
  });
}

export function
removePaddingAndResizeBack<T extends(tf.Tensor2D | tf.Tensor3D)>(
    resizedAndPadded: T, [originalHeight, originalWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]): T {
  const [height, width] = resizedAndPadded.shape;
  // remove padding that was added
  const cropH = height - (padT + padB);
  const cropW = width - (padL + padR);

  return tf.tidy(() => {
    let withPaddingRemoved: T;
    if (resizedAndPadded.rank === 2) {
      withPaddingRemoved = tf.slice2d(
                               resizedAndPadded as tf.Tensor2D, [padT, padL],
                               [cropH, cropW]) as T;
    } else {
      withPaddingRemoved = tf.slice3d(
                               resizedAndPadded as tf.Tensor3D, [padT, padL, 0],
                               [cropH, cropW, resizedAndPadded.shape[2]]) as T;
    }

    const atOriginalSize =
        resize(withPaddingRemoved, [originalHeight, originalWidth], true);

    return atOriginalSize;
  });
}

function resize<T extends(tf.Tensor2D | tf.Tensor3D)>(
    input: T, [height, width]: [number, number], nearestNeighbor?: boolean): T {
  if (input.rank === 2) {
    return resize2d(input as tf.Tensor2D, [height, width], nearestNeighbor) as
        T;
  } else {
    return (input as tf.Tensor3D)
               .resizeBilinear([height, width], nearestNeighbor) as T;
  }
}
