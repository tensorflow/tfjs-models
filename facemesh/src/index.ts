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

import * as blazeface from '@tensorflow-models/blazeface';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box} from './box';
import {MESH_ANNOTATIONS} from './keypoints';
import {BlazePipeline} from './pipeline';

const BLAZE_MESH_GRAPHMODEL_PATH =
    'https://storage.googleapis.com/learnjs-data/facemesh_staging/facemesh_faceflag-ultralite_shift30-2018_12_21-v0.hdf5_tfjs/model.json';

export async function load() {
  const faceMesh = new FaceMesh();
  await faceMesh.load();
  return faceMesh;
}

export class FaceMesh {
  private pipeline: BlazePipeline;
  private detectionConfidence: number;

  async load({
    meshWidth = 128,
    meshHeight = 128,
    maxContinuousChecks = 5,
    detectionConfidence = 0.9,
    maxFaces = 10,
    iouThreshold = 0.3,
    scoreThreshold = 0.75
  } = {}) {
    const [blazeFace, blazeMeshModel] = await Promise.all([
      this.loadFaceModel(maxFaces, iouThreshold, scoreThreshold),
      this.loadMeshModel()
    ]);

    this.pipeline = new BlazePipeline(
        blazeFace, blazeMeshModel, meshWidth, meshHeight, maxContinuousChecks,
        maxFaces);

    this.detectionConfidence = detectionConfidence;
  }

  loadFaceModel(maxFaces: number, iouThreshold: number, scoreThreshold: number):
      Promise<blazeface.BlazeFaceModel> {
    return blazeface.load({maxFaces, iouThreshold, scoreThreshold});
  }

  loadMeshModel(): Promise<tfconv.GraphModel> {
    return tfconv.loadGraphModel(BLAZE_MESH_GRAPHMODEL_PATH);
  }

  clearPipelineROIs(flag: number[][]) {
    if (flag[0][0] < this.detectionConfidence) {
      this.pipeline.clearROIs();
    }
  }

  async estimateFaces(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false): Promise<Array<{
    faceInViewConfidence: number,
    mesh: tf.Tensor2D,
    scaledMesh: tf.Tensor2D,
    boundingBox: {topLeft: tf.Tensor2D, bottomRight: tf.Tensor2D}
  }|{
    faceInViewConfidence: number,
    mesh: number[][],
    scaledMesh: number[][],
    boundingBox: {topLeft: number[], bottomRight: number[]},
    annotations: {[key: string]: number[][]}
  }>> {
    if (!(input instanceof tf.Tensor)) {
      input = tf.browser.fromPixels(input);
    }

    const inputToFloat = input.toFloat();
    const image = inputToFloat.expandDims(0) as tf.Tensor4D;
    const predictions = await this.pipeline.predict(image) as {};

    input.dispose();
    inputToFloat.dispose();

    if (predictions && (predictions as any[]).length) {
      return Promise.all((predictions as any).map(async (prediction: any) => {
        const [coords2d, coords2dScaled, landmarksBox, flag] =
            prediction as [tf.Tensor2D, tf.Tensor2D, Box, tf.Tensor2D];

        const [coordsArr, coordsArrScaled, topLeft, bottomRight, flagArr] =
            await Promise.all([
              coords2d, coords2dScaled, landmarksBox.startPoint,
              landmarksBox.endPoint, flag
            ].map(async d => await d.array()));

        flag.dispose();
        coords2dScaled.dispose();
        coords2d.dispose();

        this.clearPipelineROIs(flagArr);

        const annotations: {[key: string]: number[][]} = {};
        for (const key in MESH_ANNOTATIONS) {
          annotations[key] =
              (MESH_ANNOTATIONS[key] as number[])
                  .map((index: number): number[] => coordsArrScaled[index]) as
              number[][];
        }

        return {
          faceInViewConfidence: flagArr[0][0],
          boundingBox: {topLeft: topLeft[0], bottomRight: bottomRight[0]},
          mesh: coordsArr,
          scaledMesh: coordsArrScaled,
          annotations
        };
      })) as any;

      // if (returnTensors) {
      //   const flagArr = await flag.array();
      //   this.clearPipelineROIs(flagArr);

      //   return {
      //     faceInViewConfidence: flagArr[0][0],
      //     mesh: coords2d,
      //     scaledMesh: coords2dScaled,
      //     boundingBox: {
      //       topLeft: landmarksBox.startPoint,
      //       bottomRight: landmarksBox.endPoint
      //     }
      //   };
      // }
    }

    // No face in view.
    return null;
  }
}
