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

import * as tf from '@tensorflow/tfjs-core';

import {connectedPartIndices} from './keypoints';
import {PoseNetOutputStride} from './posenet_model';
import {Keypoint, Padding, Pose, PosenetInput, TensorBuffer3D, Vector2D} from './types';

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

  return tf.buffer(tensor.shape, type, tensorData as Float32Array) as
      tf.TensorBuffer<rank>;
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
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

export function getValidResolution(
    imageScaleFactor: number, inputDimension: number,
    outputStride: PoseNetOutputStride): number {
  const evenResolution = inputDimension * imageScaleFactor - 1;

  return evenResolution - (evenResolution % outputStride) + 1;
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
  })

  return {
    resized, padding: {top: padT, left: padL, right: padR, bottom: padB}
  }
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
