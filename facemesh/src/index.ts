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

import {MESH_ANNOTATIONS} from './keypoints';
import {Pipeline, Prediction} from './pipeline';

const BLAZE_MESH_GRAPHMODEL_PATH =
    'https://storage.googleapis.com/learnjs-data/facemesh_staging/facemesh_faceflag-ultralite_shift30-2018_12_21-v0.hdf5_tfjs/model.json';

export type AnnotatedPrediction = {
  faceInViewConfidence: number|tf.Scalar,
  boundingBox:
      {topLeft: number[][]|tf.Tensor2D, bottomRight: number[][]|tf.Tensor2D},
  mesh: number[][]|tf.Tensor2D,
  scaledMesh: number[][]|tf.Tensor2D,
  annotations?: {[key: string]: number[][]}
};

export async function load() {
  const faceMesh = new FaceMesh();
  await faceMesh.load();
  return faceMesh;
}

export class FaceMesh {
  private pipeline: Pipeline;
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

    this.pipeline = new Pipeline(
        blazeFace, blazeMeshModel, meshWidth, meshHeight, maxContinuousChecks,
        maxFaces);

    this.detectionConfidence = detectionConfidence;
  }

  static getAnnotations() {
    return MESH_ANNOTATIONS;
  }

  loadFaceModel(maxFaces: number, iouThreshold: number, scoreThreshold: number):
      Promise<blazeface.BlazeFaceModel> {
    return blazeface.load({maxFaces, iouThreshold, scoreThreshold});
  }

  loadMeshModel(): Promise<tfconv.GraphModel> {
    return tfconv.loadGraphModel(BLAZE_MESH_GRAPHMODEL_PATH);
  }

  clearPipelineROIs(flag: number) {
    if (flag < this.detectionConfidence) {
      this.pipeline.clearROIs();
    }
  }

  async estimateFaces(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false): Promise<AnnotatedPrediction[]> {
    if (!(input instanceof tf.Tensor)) {
      input = tf.browser.fromPixels(input);
    }

    const inputToFloat = input.toFloat();
    const image = inputToFloat.expandDims(0) as tf.Tensor4D;
    const predictions = await this.pipeline.predict(image) as Prediction[];

    input.dispose();
    inputToFloat.dispose();

    if (predictions && predictions.length) {
      return Promise.all(predictions.map(async (prediction: Prediction) => {
        const {coords, scaledCoords, box, flag} = prediction;
        let tensorsToRead: Array<tf.Tensor2D|tf.Scalar> = [flag];
        if (!returnTensors) {
          tensorsToRead = tensorsToRead.concat(
              [coords, scaledCoords, box.startPoint, box.endPoint]);
        }

        const tensorValues =
            await Promise.all(tensorsToRead.map(async d => await d.array()));
        const flagValue = tensorValues[0];

        flag.dispose();
        this.clearPipelineROIs(flagValue as number);

        if (returnTensors) {
          return {
            faceInViewConfidence: flag,
            mesh: coords,
            scaledMesh: scaledCoords,
            boundingBox:
                {topLeft: box.startPoint, bottomRight: box.endPoint}
          } as AnnotatedPrediction;
        }

        const [coordsArr, coordsArrScaled, topLeft, bottomRight] =
            tensorValues.slice(1);

        scaledCoords.dispose();
        coords.dispose();

        const annotations: {[key: string]: number[][]} = {};
        for (const key in MESH_ANNOTATIONS) {
          annotations[key] =
              (MESH_ANNOTATIONS[key] as number[])
                  .map(
                      (index: number): number[] =>
                          (coordsArrScaled as number[][])[index]) as number[][];
        }

        return {
          faceInViewConfidence: flagValue,
          boundingBox: {topLeft, bottomRight},
          mesh: coordsArr,
          scaledMesh: coordsArrScaled,
          annotations
        } as AnnotatedPrediction;
      }));
    }

    // No face in view.
    return null;
  }
}
