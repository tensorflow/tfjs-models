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

import {NUM_KEYPOINTS} from '../keypoints';
import {PartSegmentation, PersonSegmentation, Pose} from '../types';

function getScale(
    [height, width]: [number, number],
    [inputResolutionY, inputResolutionX]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    [number, number] {
  const scaleY = inputResolutionY / (padT + padB + height);
  const scaleX = inputResolutionX / (padL + padR + width);
  return [scaleX, scaleY];
}

export function toPersonKSegmentation(
    segmentation: tf.Tensor, k: number): tf.Tensor2D {
  return tf.tidy(
      () => (segmentation.equal(tf.scalar(k)).toInt() as tf.Tensor2D));
}

export function toPersonKPartSegmentation(
    segmentation: tf.Tensor, bodyParts: tf.Tensor, k: number): tf.Tensor2D {
  return tf.tidy(
      () => (segmentation.equal(tf.scalar(k)).toInt().mul(bodyParts.add(1)))
                .sub(1) as tf.Tensor2D);
}

function decodeMultipleMasksTensorGPU(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D,
    posesAboveScore: Pose[], height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    refineSteps: number, minKptScore: number,
    maxNumPeople: number): tf.Tensor2D {
  // The height/width of the image/canvas itself.
  const [origHeight, origWidth] = segmentation.shape;
  // The height/width of the output of the model.
  const [outHeight, outWidth] = longOffsets.shape.slice(0, 2);

  const shapedLongOffsets: tf.Tensor4D =
      longOffsets.reshape([outHeight, outWidth, 2, NUM_KEYPOINTS]);

  // Make pose tensor of shape [MAX_NUM_PEOPLE, NUM_KEYPOINTS, 3] where
  // the last 3 coordinates correspond to the score, h and w coordinate of that
  // keypoint.
  const poseVals = new Float32Array(maxNumPeople * NUM_KEYPOINTS * 3).fill(0.0);
  for (let i = 0; i < posesAboveScore.length; i++) {
    const poseOffset = i * NUM_KEYPOINTS * 3;
    const pose = posesAboveScore[i];
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const keypoint = pose.keypoints[kp];
      const offset = poseOffset + kp * 3;
      poseVals[offset] = keypoint.score;
      poseVals[offset + 1] = keypoint.position.y;
      poseVals[offset + 2] = keypoint.position.x;
    }
  }

  const [scaleX, scaleY] = getScale(
      [height, width], [inHeight, inWidth], [[padT, padB], [padL, padR]]);

  const posesTensor = tf.tensor(poseVals, [maxNumPeople, NUM_KEYPOINTS, 3]);

  const program: tf.webgl.GPGPUProgram = {
    variableNames: ['segmentation', 'longOffsets', 'poses'],
    outputShape: [origHeight, origWidth],
    userCode: `
    int convertToPositionInOutput(int pos, int pad, float scale, int stride) {
      return round(((float(pos + pad) + 1.0) * scale - 1.0) / float(stride));
    }

    float convertToPositionInOutputFloat(int pos, int pad, float scale, int stride) {
      return ((float(pos + pad) + 1.0) * scale - 1.0) / float(stride);
    }

    float dist(float x1, float y1, float x2, float y2) {
      return pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0);
    }

    float sampleLongOffsets(float h, float w, int d, int k) {
      float fh = fract(h);
      float fw = fract(w);
      int clH = int(ceil(h));
      int clW = int(ceil(w));
      int flH = int(floor(h));
      int flW = int(floor(w));
      float o11 = getLongOffsets(flH, flW, d, k);
      float o12 = getLongOffsets(flH, clW, d, k);
      float o21 = getLongOffsets(clH, flW, d, k);
      float o22 = getLongOffsets(clH, clW, d, k);
      float o1 = mix(o11, o12, fw);
      float o2 = mix(o21, o22, fw);
      return mix(o1, o2, fh);
    }

    int findNearestPose(int h, int w) {
      float prob = getSegmentation(h, w);
      if (prob < 1.0) {
        return -1;
      }

      // Done(Tyler): convert from output space h/w to strided space.
      float stridedH = convertToPositionInOutputFloat(
        h, ${padT}, ${scaleY}, ${stride});
      float stridedW = convertToPositionInOutputFloat(
        w, ${padL}, ${scaleX}, ${stride});

      float minDist = 1000000.0;
      int iMin = -1;
      for (int i = 0; i < ${maxNumPeople}; i++) {
        float curDistSum = 0.0;
        int numKpt = 0;
        for (int k = 0; k < ${NUM_KEYPOINTS}; k++) {
          float dy = sampleLongOffsets(stridedH, stridedW, 0, k);
          float dx = sampleLongOffsets(stridedH, stridedW, 1, k);

          float y = float(h) + dy;
          float x = float(w) + dx;

          for (int s = 0; s < ${refineSteps}; s++) {
            int yRounded = round(min(y, float(${height - 1.0})));
            int xRounded = round(min(x, float(${width - 1.0})));

            float yStrided = convertToPositionInOutputFloat(
              yRounded, ${padT}, ${scaleY}, ${stride});
            float xStrided = convertToPositionInOutputFloat(
              xRounded, ${padL}, ${scaleX}, ${stride});

            float dy = sampleLongOffsets(yStrided, xStrided, 0, k);
            float dx = sampleLongOffsets(yStrided, xStrided, 1, k);

            y = y + dy;
            x = x + dx;
          }

          float poseScore = getPoses(i, k, 0);
          float poseY = getPoses(i, k, 1);
          float poseX = getPoses(i, k, 2);
          if (poseScore > ${minKptScore}) {
            numKpt = numKpt + 1;
            curDistSum = curDistSum + dist(x, y, poseX, poseY);
          }
        }
        if (numKpt > 0 && curDistSum / float(numKpt) < minDist) {
          minDist = curDistSum / float(numKpt);
          iMin = i;
        }
      }
      return iMin;
    }

    void main() {
        ivec2 coords = getOutputCoords();
        int nearestPose = findNearestPose(coords[0], coords[1]);
        setOutput(float(nearestPose));
      }
  `
  };
  const webglBackend = tf.backend() as tf.webgl.MathBackendWebGL;
  const result = webglBackend.compileAndRun(
                     program, [segmentation, shapedLongOffsets, posesTensor]) as
      tf.Tensor2D;

  return result;
}

interface Pair {
  x: number;
  y: number;
}

function computeDistance(embedding: Pair[], pose: Pose, minPartScore = 0.3) {
  let distance = 0.0;
  let numKpt = 0;
  for (let p = 0; p < embedding.length; p++) {
    if (pose.keypoints[p].score > minPartScore) {
      numKpt += 1;
      distance += (embedding[p].x - pose.keypoints[p].position.x) ** 2 +
          (embedding[p].y - pose.keypoints[p].position.y) ** 2;
    }
  }
  if (numKpt === 0) {
    distance = Infinity;
  } else {
    distance = distance / numKpt;
  }
  return distance;
}

function convertToPositionInOuput(
    position: Pair, [padT, padL]: [number, number],
    [scaleX, scaleY]: [number, number], stride: number): Pair {
  const y = Math.round(((padT + position.y + 1.0) * scaleY - 1.0) / stride);
  const x = Math.round(((padL + position.x + 1.0) * scaleX - 1.0) / stride);
  return {x, y};
}

function getEmbedding(
    location: Pair, keypointIndex: number,
    convertToPosition: (pair: Pair) => Pair, outputResolutionX: number,
    longOffsets: Float32Array, refineSteps: number,
    [height, width]: [number, number]): Pair {
  const newLocation = convertToPosition(location);

  const nn = newLocation.y * outputResolutionX + newLocation.x;
  let dy = longOffsets[NUM_KEYPOINTS * (2 * nn) + keypointIndex];
  let dx = longOffsets[NUM_KEYPOINTS * (2 * nn + 1) + keypointIndex];
  let y = location.y + dy;
  let x = location.x + dx;
  for (let t = 0; t < refineSteps; t++) {
    y = Math.min(y, height - 1);
    x = Math.min(x, width - 1);
    const newPos = convertToPosition({x, y});
    const nn = newPos.y * outputResolutionX + newPos.x;
    dy = longOffsets[NUM_KEYPOINTS * (2 * nn) + keypointIndex];
    dx = longOffsets[NUM_KEYPOINTS * (2 * nn + 1) + keypointIndex];
    y = y + dy;
    x = x + dx;
  }

  return {x, y};
}

function matchEmbeddingToInstance(
    location: Pair, longOffsets: Float32Array, poses: Pose[],
    numKptForMatching: number, [padT, padL]: [number, number],
    [scaleX, scaleY]: [number, number], outputResolutionX: number,
    [height, width]: [number, number], stride: number,
    refineSteps: number): number {
  const embed: Pair[] = [];
  const convertToPosition = (pair: Pair) =>
      convertToPositionInOuput(pair, [padT, padL], [scaleX, scaleY], stride);

  for (let keypointsIndex = 0; keypointsIndex < numKptForMatching;
       keypointsIndex++) {
    const embedding = getEmbedding(
        location, keypointsIndex, convertToPosition, outputResolutionX,
        longOffsets, refineSteps, [height, width]);

    embed.push(embedding);
  }

  let kMin = -1;
  let kMinDist = Infinity;
  for (let k = 0; k < poses.length; k++) {
    const dist = computeDistance(embed, poses[k]);
    if (dist < kMinDist) {
      kMin = k;
      kMinDist = dist;
    }
  }
  return kMin;
}

export function getOutputResolution(
    [inputResolutionY, inputResolutionX]: [number, number],
    stride: number): [number, number] {
  const outputResolutionX = Math.round((inputResolutionX - 1.0) / stride + 1.0);
  const outputResolutionY = Math.round((inputResolutionY - 1.0) / stride + 1.0);
  return [outputResolutionX, outputResolutionY];
}

export function decodeMultipleMasksTensorCPU(
    segmentation: Uint8Array, longOffsets: Float32Array,
    posesAboveScore: Pose[], height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    refineSteps: number, numKptForMatching = 5): Uint8Array[] {
  const dataArrays =
      posesAboveScore.map(x => new Uint8Array(height * width).fill(0));

  const [scaleX, scaleY] = getScale(
      [height, width], [inHeight, inWidth], [[padT, padB], [padL, padR]]);
  const [outputResolutionX, ] =
    getOutputResolution([inHeight, inWidth], stride);
  for (let i = 0; i < height; i += 1) {
    for (let j = 0; j < width; j += 1) {
      const n = i * width + j;
      const prob = segmentation[n];
      if (prob === 1) {
        const kMin = matchEmbeddingToInstance(
            {x: j, y: i}, longOffsets, posesAboveScore, numKptForMatching,
            [padT, padL], [scaleX, scaleY], outputResolutionX, [height, width],
            stride, refineSteps);
        if (kMin >= 0) {
          dataArrays[kMin][n] = 1;
        }
      }
    }
  }

  return dataArrays;
}

export function decodeMultiplePartMasksTensorCPU(
    segmentation: Uint8Array, longOffsets: Float32Array,
    partSegmentaion: Uint8Array, posesAboveScore: Pose[], height: number,
    width: number, stride: number, [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    refineSteps: number, numKptForMatching = 5): Int32Array[] {
  const dataArrays =
      posesAboveScore.map(x => new Int32Array(height * width).fill(-1));

  const [scaleX, scaleY] = getScale(
      [height, width], [inHeight, inWidth], [[padT, padB], [padL, padR]]);
  const [outputResolutionX, ] =
    getOutputResolution([inHeight, inWidth], stride);

  for (let i = 0; i < height; i += 1) {
    for (let j = 0; j < width; j += 1) {
      const n = i * width + j;
      const prob = segmentation[n];
      if (prob === 1) {
        const kMin = matchEmbeddingToInstance(
            {x: j, y: i}, longOffsets, posesAboveScore, numKptForMatching,
            [padT, padL], [scaleX, scaleY], outputResolutionX, [height, width],
            stride, refineSteps);
        if (kMin >= 0) {
          dataArrays[kMin][n] = partSegmentaion[n];
        }
      }
    }
  }

  return dataArrays;
}

function isWebGlBackend() {
  return getBackend() === 'webgl';
}

export async function decodePersonInstanceMasks(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D, poses: Pose[],
    height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    minPoseScore = 0.2, refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PersonSegmentation[]> {
  // Filter out poses with smaller score.
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let personSegmentationsData: Uint8Array[];

  if (isWebGlBackend()) {
    const personSegmentations = tf.tidy(() => {
      const masksTensor = decodeMultipleMasksTensorGPU(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps,
          minKeypointScore, maxNumPeople);

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

    personSegmentationsData = decodeMultipleMasksTensorCPU(
        segmentationsData, longOffsetsData, posesAboveScore, height, width,
        stride, [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps);
  }

  return personSegmentationsData.map(
      (data, i) => ({data, pose: posesAboveScore[i], width, height}));
}

export async function decodePersonInstancePartMasks(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D,
    partSegmentation: tf.Tensor, poses: Pose[], height: number, width: number,
    stride: number, [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    minPoseScore = 0.2, refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PartSegmentation[]> {
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let partSegmentationsByPersonData: Int32Array[];

  if (isWebGlBackend()) {
    const partSegmentations = tf.tidy(() => {
      const masksTensor = decodeMultipleMasksTensorGPU(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps,
          minKeypointScore, maxNumPeople);

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

    partSegmentationsByPersonData = decodeMultiplePartMasksTensorCPU(
        segmentationsData, longOffsetsData, partSegmentaionData,
        posesAboveScore, height, width, stride, [inHeight, inWidth],
        [[padT, padB], [padL, padR]], refineSteps);
  }

  return partSegmentationsByPersonData.map(
      (data, k) => ({pose: posesAboveScore[k], data, height, width}));
}
