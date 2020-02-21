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

// TODO: CHANGE TO TFHUB LINK ONCE AVAILABLE.
const BLAZE_MESH_GRAPHMODEL_PATH =
    'https://storage.googleapis.com/learnjs-data/facemesh_staging/facemesh_facecontours_faceflag-blaze_shift30-2019_01_14-v0.hdf5_tfjs_fixed_batch/model.json';

// const BLAZE_MESH_GRAPHMODEL_PATH =
//     'https://storage.googleapis.com/learnjs-data/facemesh_staging/facemesh_faceflag-ultralite_shift30-2018_12_21-v0.hdf5_tfjs/model.json';

export type AnnotatedPrediction = {
  faceInViewConfidence: number|tf.Scalar,
  boundingBox: {
    topLeft: [number, number]|tf.Tensor1D,
    bottomRight: [number, number]|tf.Tensor1D
  },
  mesh: number[][]|tf.Tensor2D,
  scaledMesh: number[][]|tf.Tensor2D,
  /*Annotated keypoints. */
  annotations?: {[key: string]: number[][]}
};

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipFaceHorizontal(
    face: AnnotatedPrediction, imageWidth: number,
    return3d: boolean): AnnotatedPrediction {
  if (face.mesh instanceof tf.Tensor &&
      face.boundingBox.topLeft instanceof tf.Tensor &&
      face.boundingBox.bottomRight instanceof tf.Tensor) {
    const subtractBasis =
        return3d ? [imageWidth - 1, 0, 0] : [imageWidth - 1, 0];
    const multiplyBasis = return3d ? [1, -1, 1] : [1, -1];

    return Object.assign({}, face, {
      boundingBox: {
        topLeft: tf.concat([
          tf.sub(imageWidth - 1, face.boundingBox.topLeft.slice(0, 1)),
          face.boundingBox.topLeft.slice(1, 1)
        ]),
        bottomRight: tf.concat([
          tf.sub(imageWidth - 1, face.boundingBox.bottomRight.slice(0, 1)),
          face.boundingBox.bottomRight.slice(1, 1)
        ])
      },
      mesh: tf.sub(tf.tensor1d(subtractBasis), face.mesh)
                .mul(tf.tensor1d(multiplyBasis)) as tf.Tensor2D,
      scaledMesh: tf.sub(tf.tensor1d(subtractBasis), face.scaledMesh)
                      .mul(tf.tensor1d(multiplyBasis)) as tf.Tensor2D
    });
  }

  return Object.assign({}, face, {
    boundingBox: {
      topLeft: [
        imageWidth - 1 - (face.boundingBox.topLeft as [number, number])[0],
        (face.boundingBox.topLeft as [number, number])[1]
      ],
      bottomRight: [
        imageWidth - 1 - (face.boundingBox.bottomRight as [number, number])[0],
        (face.boundingBox.bottomRight as [number, number])[1]
      ]
    },
    mesh: (face.mesh as number[][])
              .map((coord: [number, number]|[number, number, number]) => {
                const flippedCoord = coord.slice(0);
                flippedCoord[0] = imageWidth - 1 - coord[0];
                return flippedCoord;
              }),
    scaledMesh: (face.scaledMesh as number[][])
                    .map((coord: [number, number]|[number, number, number]) => {
                      const flippedCoord = coord.slice(0);
                      flippedCoord[0] = imageWidth - 1 - coord[0];
                      return flippedCoord;
                    })
  });
}

export async function load(options = {}) {
  const faceMesh = new FaceMesh();
  await faceMesh.load(options);
  return faceMesh;
}

export class FaceMesh {
  private pipeline: Pipeline;
  private detectionConfidence: number;

  async load({
    meshWidth = 192,
    meshHeight = 192,
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
      returnTensors = false, flipHorizontal = false,
      return3d = false): Promise<AnnotatedPrediction[]> {
    if (!(input instanceof tf.Tensor)) {
      input = tf.browser.fromPixels(input);
    }

    const [, width] = getInputTensorDimensions(input);
    const inputToFloat = input.toFloat();
    const image = inputToFloat.expandDims(0) as tf.Tensor4D;
    const predictions =
        await this.pipeline.predict(image, return3d) as Prediction[];

    input.dispose();
    inputToFloat.dispose();
    image.dispose();

    if (predictions && predictions.length) {
      return Promise
          .all(
              predictions.map(
                  async (prediction: Prediction) => {
                    const {coords, scaledCoords, box, flag} = prediction;
                    let tensorsToRead: tf.Tensor[] = [flag];
                    if (!returnTensors) {
                      tensorsToRead = tensorsToRead.concat(
                          [coords, scaledCoords, box.startPoint, box.endPoint]);
                    }

                    const tensorValues = await Promise.all(tensorsToRead.map(
                        async (d: tf.Tensor) => await d.array()));
                    const flagValue = tensorValues[0] as number;

                    flag.dispose();
                    this.clearPipelineROIs(flagValue);

                    if (returnTensors) {
                      let annotatedPrediction = {
                        faceInViewConfidence: flag,
                        mesh: coords,
                        scaledMesh: scaledCoords,
                        boundingBox: {
                          topLeft: box.startPoint.squeeze(),
                          bottomRight: box.endPoint.squeeze()
                        }
                      } as AnnotatedPrediction;

                      if (flipHorizontal) {
                        annotatedPrediction = flipFaceHorizontal(
                            annotatedPrediction, width, return3d);
                      }

                      return annotatedPrediction;
                    }

                    const [coordsArr, coordsArrScaled, topLeft, bottomRight] =
                      tensorValues.slice(1) as [
                        number[][],
                        number[][],
                        [number, number],
                        [number, number]];

                    scaledCoords.dispose();
                    coords.dispose();

                    let annotatedPrediction: AnnotatedPrediction = {
                      faceInViewConfidence: flagValue,
                      boundingBox: {
                        topLeft: topLeft as [number, number],
                        bottomRight: bottomRight as [number, number]
                      },
                      mesh: coordsArr as number[][],
                      scaledMesh: coordsArrScaled as number[][]
                    };

                    if (flipHorizontal) {
                      annotatedPrediction = flipFaceHorizontal(
                          annotatedPrediction, width, return3d);
                    }

                    const annotations: {[key: string]: number[][]} = {};
                    for (const key in MESH_ANNOTATIONS) {
                      annotations[key] =
                          (MESH_ANNOTATIONS[key] as number[])
                              .map(
                                  (index: number): number[] =>
                                      (annotatedPrediction.scaledMesh as
                                       number[][])[index]) as number[][];
                    }
                    annotatedPrediction['annotations'] = annotations;

                    return annotatedPrediction;
                  }));
    }

    return null;
  }
}
