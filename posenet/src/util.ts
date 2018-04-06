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

import {Keypoint} from '.';
import {connectedJointIndeces} from './keypoints';
import {TensorBuffer3D, Vector2D} from './types';

function eitherPointDoesntMeetConfidence(
    a: number, b: number, minConfidence: number) {
  return (a < minConfidence || b < minConfidence);
}

export function getAdjacentKeyPoints(
    keypoints: Keypoint[], minConfidence: number): Keypoint[][] {
  return connectedJointIndeces.reduce(
      (result: Keypoint[][], [leftJoint, rightJoint]): Keypoint[][] => {
        if (eitherPointDoesntMeetConfidence(
                keypoints[leftJoint].score, keypoints[rightJoint].score,
                minConfidence)) {
          return result;
        }

        const leftPoint = keypoints[leftJoint];
        const rightPoint = keypoints[rightJoint];

        result.push([leftPoint, rightPoint]);

        return result;
      }, []);
}

export function setHeatmapAsAlphaChannel(
    imagePixels: tf.Tensor3D, outputStride: number,
    heatmapImage: tf.Tensor2D): tf.Tensor3D {
  const [height, width] = imagePixels.shape;

  return tf.tidy(() => {
    const scaledUp = resizeBilinearGrayscale(heatmapImage, [
      heatmapImage.shape[0] * outputStride, heatmapImage.shape[1] * outputStride
    ]);

    const rgb =
        imagePixels.slice([0, 0, 0], [height, width, 3]).div(tf.scalar(255)) as
        tf.Tensor3D;
    const a = scaledUp.slice([0, 0, 0], [height, width, 1]);

    const result = tf.concat3d([rgb, a], 2);

    return result;
  })
}

export function toHeatmapImage(heatmapScores: tf.Tensor3D): tf.Tensor2D {
  return tf.tidy(() => {
    return heatmapScores.sum(2).minimum(tf.scalar(1)) as tf.Tensor2D;
  });
}

export function resizeBilinearGrayscale(
    heatmapImage: tf.Tensor2D, size: [number, number]) {
  return tf.tidy(() => {
    const channel = heatmapImage.expandDims(2) as tf.Tensor3D;
    const rgb = tf.concat([channel, channel, channel], 2) as tf.Tensor3D;
    return tf.image.resizeBilinear(rgb, size);
  })
}

export function toSingleChannelPixels(tensor: tf.Tensor2D) {
  return new ImageData(
      new Uint8ClampedArray(tensor.mul(tf.scalar(255)).toInt().buffer().values),
      tensor.shape[1], tensor.shape[0]);
}

const {NEGATIVE_INFINITY, POSITIVE_INFINITY} = Number;
export function getBoundingBox(keypoints: Keypoint[]) {
  return keypoints.reduce(({maxX, maxY, minX, minY}, {point: {x, y}}) => {
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
