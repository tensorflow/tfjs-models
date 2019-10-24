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

import {NUM_KEYPOINTS} from '../keypoints';
import {Padding, Pose} from '../types';
import {getScale} from './util';

export function decodeMultipleMasksWebGl(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D,
    posesAboveScore: Pose[], height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number], padding: Padding,
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

  const [scaleX, scaleY] =
      getScale([height, width], [inHeight, inWidth], padding);

  const posesTensor = tf.tensor(poseVals, [maxNumPeople, NUM_KEYPOINTS, 3]);

  const {top: padT, left: padL} = padding;

  const program: tf.webgl.GPGPUProgram = {
    variableNames: ['segmentation', 'longOffsets', 'poses'],
    outputShape: [origHeight, origWidth],
    userCode: `
    int convertToPositionInOutput(int pos, int pad, float scale, int stride) {
      return round(((float(pos + pad) + 1.0) * scale - 1.0) / float(stride));
    }

    float convertToPositionInOutputFloat(
        int pos, int pad, float scale, int stride) {
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
  return webglBackend.compileAndRun(
      program, [segmentation, shapedLongOffsets, posesTensor]);
}
