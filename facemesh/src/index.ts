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
const MESH_MODEL_INPUT_WIDTH = 192;
const MESH_MODEL_INPUT_HEIGHT = 192;

export type AnnotatedPrediction = {
  faceInViewConfidence: number|tf.Scalar,
  boundingBox: {
    topLeft: [number, number]|tf.Tensor1D,
    bottomRight: [number, number]|tf.Tensor1D
  },
  mesh: Array<[number, number, number]>|tf.Tensor2D,
  scaledMesh: Array<[number, number, number]>|tf.Tensor2D,
  /*Annotated keypoints. Not available if returning tensors. */
  annotations?: {[key: string]: Array<[number, number, number]>}
};

/**
 * Load the model.
 * @param options - a configuration object with the following properties:
 *  `maxContinuousChecks` How many frames to go without running the bounding box
 * detector. Only relevant if maxFaces > 1. Defaults to 5.
 *  `detectionConfidence` Threshold for discarding a prediction. Defaults to
 * 0.9.
 *  `maxFaces` The maximum number of faces detected in the input. Should be
 * set to the minimum number for performance. Defaults to 10.
 *  `iouThreshold` A float representing the threshold for deciding whether boxes
 * overlap too much in non-maximum suppression. Must be between [0, 1]. Defaults
 * to 0.3.
 *  `scoreThreshold` A threshold for deciding when to remove boxes based
 * on score in non-maximum suppression. Defaults to 0.75.
 */
export async function load({
  maxContinuousChecks = 5,
  detectionConfidence = 0.9,
  maxFaces = 10,
  iouThreshold = 0.3,
  scoreThreshold = 0.75
} = {}) {
  const faceMesh = new FaceMesh();

  await faceMesh.load(
      maxContinuousChecks, detectionConfidence, maxFaces, iouThreshold,
      scoreThreshold);
  return faceMesh;
}

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipFaceHorizontal(
    face: AnnotatedPrediction, imageWidth: number): AnnotatedPrediction {
  if (face.mesh instanceof tf.Tensor) {
    const [topLeft, bottomRight, mesh, scaledMesh] = tf.tidy(() => {
      const subtractBasis = tf.tensor1d([imageWidth - 1, 0, 0]);
      const multiplyBasis = tf.tensor1d([1, -1, 1]);

      return [
        tf.concat([
          tf.sub(
              imageWidth - 1,
              (face.boundingBox.topLeft as tf.Tensor1D).slice(0, 1)),
          (face.boundingBox.topLeft as tf.Tensor1D).slice(1, 1)
        ]),
        tf.concat([
          tf.sub(
              imageWidth - 1,
              (face.boundingBox.bottomRight as tf.Tensor1D).slice(0, 1)),
          (face.boundingBox.bottomRight as tf.Tensor1D).slice(1, 1)
        ]),
        tf.sub(subtractBasis, face.mesh).mul(multiplyBasis),
        tf.sub(subtractBasis, face.scaledMesh).mul(multiplyBasis)
      ];
    });

    return Object.assign(
        {}, face, {boundingBox: {topLeft, bottomRight}, mesh, scaledMesh});
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
    mesh: (face.mesh as Array<[number, number, number]>).map(coord => {
      const flippedCoord = coord.slice(0);
      flippedCoord[0] = imageWidth - 1 - coord[0];
      return flippedCoord;
    }),
    scaledMesh:
        (face.scaledMesh as Array<[number, number, number]>).map(coord => {
          const flippedCoord = coord.slice(0);
          flippedCoord[0] = imageWidth - 1 - coord[0];
          return flippedCoord;
        })
  });
}

export class FaceMesh {
  private pipeline: Pipeline;
  private detectionConfidence: number;

  async load(
      maxContinuousChecks: number, detectionConfidence: number,
      maxFaces: number, iouThreshold: number, scoreThreshold: number) {
    const [blazeFace, blazeMeshModel] = await Promise.all([
      this.loadFaceModel(maxFaces, iouThreshold, scoreThreshold),
      this.loadMeshModel()
    ]);

    this.pipeline = new Pipeline(
        blazeFace, blazeMeshModel, MESH_MODEL_INPUT_WIDTH,
        MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, maxFaces);

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

  async estimateFaces(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false,
      flipHorizontal = false): Promise<AnnotatedPrediction[]> {
    if (!(input instanceof tf.Tensor)) {
      input = tf.browser.fromPixels(input);
    }

    const [, width] = getInputTensorDimensions(input);
    const inputToFloat = input.toFloat();
    const image = inputToFloat.expandDims(0) as tf.Tensor4D;

    const savedWebglPackDepthwiseConvFlag =
        tf.env().get('WEBGL_PACK_DEPTHWISECONV');
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
    const predictions = await this.pipeline.predict(image) as Prediction[];
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);

    input.dispose();
    inputToFloat.dispose();
    image.dispose();

    if (predictions && predictions.length) {
      return Promise
          .all(
              predictions.map(
                  async (prediction: Prediction, i) => {
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
                    if (flagValue < this.detectionConfidence) {
                      this.pipeline.clearRegionOfInterest(i);
                    }

                    if (returnTensors) {
                      const annotatedPrediction = {
                        faceInViewConfidence: flag,
                        mesh: coords,
                        scaledMesh: scaledCoords,
                        boundingBox: {
                          topLeft: box.startPoint.squeeze(),
                          bottomRight: box.endPoint.squeeze()
                        }
                      } as AnnotatedPrediction;

                      if (flipHorizontal) {
                        const flipped =
                            flipFaceHorizontal(annotatedPrediction, width);

                        (annotatedPrediction.mesh as tf.Tensor2D).dispose();
                        (annotatedPrediction.scaledMesh as tf.Tensor2D)
                            .dispose();
                        (annotatedPrediction.boundingBox.topLeft as tf.Tensor1D)
                            .dispose();
                        (annotatedPrediction.boundingBox.bottomRight as
                         tf.Tensor1D)
                            .dispose();

                        return flipped;
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
                      mesh: coordsArr as Array<[number, number, number]>,
                      scaledMesh: coordsArrScaled as
                          Array<[number, number, number]>
                    };

                    if (flipHorizontal) {
                      annotatedPrediction =
                          flipFaceHorizontal(annotatedPrediction, width);
                    }

                    const annotations:
                        {[key: string]: Array<[number, number, number]>} = {};
                    for (const key in MESH_ANNOTATIONS) {
                      annotations[key] =
                          (MESH_ANNOTATIONS[key] as number[])
                              .map(
                                  (index: number): number[] =>
                                      (annotatedPrediction.scaledMesh as Array<
                                           [number, number, number]>)[index]) as
                          Array<[number, number, number]>;
                    }
                    annotatedPrediction['annotations'] = annotations;

                    return annotatedPrediction;
                  }));
    }

    return null;
  }
}
