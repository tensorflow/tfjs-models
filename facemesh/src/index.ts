/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// import * as tfconv from '@tensorflow/tfjs-converter';
import './layers';

import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import {Box} from './box';
import {BlazeFaceModel} from './face';
import {BlazePipeline} from './pipeline';

const BLAZEFACE_MODEL_URL =
    'https://facemesh.s3.amazonaws.com/facedetector/rewritten_detector.json';

const BLAZE_MESH_MODEL_PATH =
    'https://facemesh.s3.amazonaws.com/facemesh/model.json';

export async function load() {
  const faceMesh = new FaceMesh();
  await faceMesh.load();
  return faceMesh;
}

const MESH_ANNOTATIONS: {[key: string]: number[]} = {
  silhouette: [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ],

  lipsUpperOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  lipsLowerOuter: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  lipsUpperInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  lipsLowerInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  rightEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
  rightEyeLower0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
  rightEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
  rightEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
  rightEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
  rightEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
  rightEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],

  rightEyebrowUpper: [156, 70, 63, 105, 66, 107, 55, 193],
  rightEyebrowLower: [35, 124, 46, 53, 52, 65],

  leftEyeUpper0: [466, 388, 387, 386, 385, 384, 398],
  leftEyeLower0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
  leftEyeUpper1: [467, 260, 259, 257, 258, 286, 414],
  leftEyeLower1: [359, 255, 339, 254, 253, 252, 256, 341, 463],
  leftEyeUpper2: [342, 445, 444, 443, 442, 441, 413],
  leftEyeLower2: [446, 261, 448, 449, 450, 451, 452, 453, 464],
  leftEyeLower3: [372, 340, 346, 347, 348, 349, 350, 357, 465],

  leftEyebrowUpper: [383, 300, 293, 334, 296, 336, 285, 417],
  leftEyebrowLower: [265, 353, 276, 283, 282, 295],

  midwayBetweenEyes: [168],

  noseTip: [1],
  noseBottom: [2],
  noseRightCorner: [98],
  noseLeftCorner: [327],

  rightCheek: [205],
  leftCheek: [425]
};

export class FaceMesh {
  private pipeline: BlazePipeline;
  private detectionConfidence: number;

  async load(
      meshWidth = 128, meshHeight = 128, maxContinuousChecks = 5,
      detectionConfidence = 0.9) {
    const [blazeFaceModel, blazeMeshModel] =
        await Promise.all([this.loadFaceModel(), this.loadMeshModel()]);

    const blazeface = new BlazeFaceModel(blazeFaceModel, meshWidth, meshHeight);

    this.pipeline = new BlazePipeline(
        blazeface, blazeMeshModel, meshWidth, meshHeight, maxContinuousChecks);

    this.detectionConfidence = detectionConfidence;
  }

  loadFaceModel(): Promise<tfl.LayersModel> {
    return tfl.loadLayersModel(BLAZEFACE_MODEL_URL);
  }

  loadMeshModel(): Promise<tfl.LayersModel> {
    return tfl.loadLayersModel(BLAZE_MESH_MODEL_PATH);
  }

  clearPipelineROIs(flag: number[][]) {
    if (flag[0][0] < this.detectionConfidence) {
      this.pipeline.clearROIs();
    }
  }

  async estimateFace(video: HTMLVideoElement, returnTensors = false): Promise<{
    faceInViewConfidence: number,
    mesh: tf.Tensor2D,
    boundingBox: {topLeft: tf.Tensor2D, bottomRight: tf.Tensor2D}
  }|{
    faceInViewConfidence: number,
    mesh: number[][],
    boundingBox: {topLeft: number[], bottomRight: number[]},
    annotations: {[key: string]: number[][]}
  }> {
    const prediction = tf.tidy(() => {
      const image =
          tf.browser.fromPixels(video).toFloat().expandDims(0) as tf.Tensor4D;
      return this.pipeline.predict(image) as {};
    });

    if (prediction != null) {
      const [coords2dScaled, landmarksBox, flag] =
          prediction as [tf.Tensor2D, Box, tf.Tensor2D];

      if (returnTensors) {
        const flagArr = await flag.array();
        this.clearPipelineROIs(flagArr);

        return {
          faceInViewConfidence: flagArr[0][0],
          mesh: coords2dScaled,
          boundingBox: {
            topLeft: landmarksBox.startPoint,
            bottomRight: landmarksBox.endPoint
          }
        };
      }

      const [coordsArr, topLeft, bottomRight, flagArr] = await Promise.all([
        coords2dScaled, landmarksBox.startPoint, landmarksBox.endPoint, flag
      ].map(async d => await d.array()));

      flag.dispose();
      coords2dScaled.dispose();

      this.clearPipelineROIs(flagArr);

      const annotations: {[key: string]: number[][]} = {};
      for (const key in MESH_ANNOTATIONS) {
        annotations[key] =
            (MESH_ANNOTATIONS[key] as number[])
                .map((index: number): number[] => coordsArr[index]) as
            number[][];
      }

      return {
        faceInViewConfidence: flagArr[0][0],
        boundingBox: {topLeft: topLeft[0], bottomRight: bottomRight[0]},
        mesh: coordsArr,
        annotations
      };
    }

    // No face in view.
    return null;
  }
}
