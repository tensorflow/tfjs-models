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

import './layers';

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {BlazeFaceModel} from './face';

const BLAZEFACE_MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/facemesh_staging/facedetector_tfjs/model.json';

export async function load(maxFaces = 10, meshWidth = 128, meshHeight = 128) {
  const faceMesh = new FaceMesh();
  await faceMesh.load(maxFaces, meshWidth, meshHeight);
  return faceMesh;
}

type FaceBoundingBox = [[number, number], [number, number]];

export class FaceMesh {
  private blazeface: BlazeFaceModel;

  async load(maxFaces: number, meshWidth: number, meshHeight: number) {
    const blazeFaceModel = await this.loadFaceModel();

    this.blazeface =
        new BlazeFaceModel(blazeFaceModel, meshWidth, meshHeight, maxFaces);
  }

  loadFaceModel(): Promise<tfconv.GraphModel> {
    return tfconv.loadGraphModel(BLAZEFACE_MODEL_URL);
  }

  async estimateFace(video: HTMLVideoElement): Promise<FaceBoundingBox[]> {
    const prediction = tf.tidy(() => {
      const image =
          tf.browser.fromPixels(video).toFloat().expandDims(0) as tf.Tensor4D;
      return this.blazeface.getSingleBoundingBox(image as tf.Tensor4D);
    });

    if (prediction != null) {
      const coords = await Promise.all(
          prediction.map(async (d: tf.Tensor1D) => await d.array()));

      return coords.map(
          arr => [arr.slice(0, 2), arr.slice(2)] as FaceBoundingBox);
    }

    return null;
  }
}
