/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import {getBackend} from '@tensorflow/tfjs-core';

import {Padding, PartSegmentation, PersonSegmentation, Pose} from '../types';

import {decodeMultipleMasksCPU, decodeMultiplePartMasksCPU} from './decode_multiple_masks_cpu';
import {decodeMultipleMasksWebGl} from './decode_multiple_masks_webgl';

export function toPersonKSegmentation(
    segmentation: tf.Tensor2D, k: number): tf.Tensor2D {
  return tf.tidy(
      () => (tf.cast(tf.equal(
          segmentation, tf.scalar(k)), 'int32') as tf.Tensor2D));
}

export function toPersonKPartSegmentation(
    segmentation: tf.Tensor2D, bodyParts: tf.Tensor2D, k: number): tf.Tensor2D {
  return tf.tidy(
      () => tf.sub(tf.mul(tf.cast(tf.equal(
          segmentation, tf.scalar(k)), 'int32'), tf.add(bodyParts, 1)), 1));
}

function isWebGlBackend() {
  return getBackend() === 'webgl';
}

export async function decodePersonInstanceMasks(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D, poses: Pose[],
    height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number], padding: Padding, minPoseScore = 0.2,
    refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PersonSegmentation[]> {
  // Filter out poses with smaller score.
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let personSegmentationsData: Uint8Array[];

  if (isWebGlBackend()) {
    const personSegmentations = tf.tidy(() => {
      const masksTensorInfo = decodeMultipleMasksWebGl(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], padding, refineSteps, minKeypointScore,
          maxNumPeople);
      const masksTensor = tf.engine().makeTensorFromDataId(
          masksTensorInfo.dataId, masksTensorInfo.shape,
          masksTensorInfo.dtype) as tf.Tensor2D;

      return posesAboveScore.map(
          (_, k) => toPersonKSegmentation(masksTensor, k));
    });

    personSegmentationsData =
        (await Promise.all(personSegmentations.map(mask => mask.data())) as
         Uint8Array[]);

    personSegmentations.forEach(x => x.dispose());
  } else {
    const segmentationsData = await segmentation.data() as Uint8Array;
    const longOffsetsData = await longOffsets.data() as Float32Array;

    personSegmentationsData = decodeMultipleMasksCPU(
        segmentationsData, longOffsetsData, posesAboveScore, height, width,
        stride, [inHeight, inWidth], padding, refineSteps);
  }

  return personSegmentationsData.map(
      (data, i) => ({data, pose: posesAboveScore[i], width, height}));
}

export async function decodePersonInstancePartMasks(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D,
    partSegmentation: tf.Tensor2D, poses: Pose[], height: number, width: number,
    stride: number, [inHeight, inWidth]: [number, number], padding: Padding,
    minPoseScore = 0.2, refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PartSegmentation[]> {
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let partSegmentationsByPersonData: Int32Array[];

  if (isWebGlBackend()) {
    const partSegmentations = tf.tidy(() => {
      const masksTensorInfo = decodeMultipleMasksWebGl(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], padding, refineSteps, minKeypointScore,
          maxNumPeople);
      const masksTensor = tf.engine().makeTensorFromDataId(
        masksTensorInfo.dataId, masksTensorInfo.shape,
        masksTensorInfo.dtype) as tf.Tensor2D;

      return posesAboveScore.map(
          (_, k) =>
              toPersonKPartSegmentation(masksTensor, partSegmentation, k));
    });

    partSegmentationsByPersonData =
        (await Promise.all(partSegmentations.map(x => x.data()))) as
        Int32Array[];

    partSegmentations.forEach(x => x.dispose());
  } else {
    const segmentationsData = await segmentation.data() as Uint8Array;
    const longOffsetsData = await longOffsets.data() as Float32Array;
    const partSegmentaionData = await partSegmentation.data() as Uint8Array;

    partSegmentationsByPersonData = decodeMultiplePartMasksCPU(
        segmentationsData, longOffsetsData, partSegmentaionData,
        posesAboveScore, height, width, stride, [inHeight, inWidth], padding,
        refineSteps);
  }

  return partSegmentationsByPersonData.map(
      (data, k) => ({pose: posesAboveScore[k], data, height, width}));
}
