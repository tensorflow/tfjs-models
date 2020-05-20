/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {UV_COORDS} from './uv_coords';

const FACEMESH_GRAPHMODEL_PATH =
    'https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1';
const MESH_MODEL_INPUT_WIDTH = 192;
const MESH_MODEL_INPUT_HEIGHT = 192;

interface AnnotatedPredictionValues {
  /** Probability of the face detection. */
  faceInViewConfidence: number;
  boundingBox: {
    /** The upper left-hand corner of the face. */
    topLeft: [number, number],
    /** The lower right-hand corner of the face. */
    bottomRight: [number, number]
  };
  /** Facial landmark coordinates. */
  mesh: Array<[number, number, number]>;
  /** Facial landmark coordinates normalized to input dimensions. */
  scaledMesh: Array<[number, number, number]>;
  /** Annotated keypoints. */
  annotations?: {[key: string]: Array<[number, number, number]>};
}

interface AnnotatedPredictionTensors {
  faceInViewConfidence: number;
  boundingBox: {topLeft: tf.Tensor1D, bottomRight: tf.Tensor1D};
  mesh: tf.Tensor2D;
  scaledMesh: tf.Tensor2D;
}

// The object returned by facemesh describing a face found in the input.
export type AnnotatedPrediction =
    AnnotatedPredictionValues|AnnotatedPredictionTensors;

/**
 * Load the model.
 *
 * @param options - a configuration object with the following properties:
 *  - `maxContinuousChecks` How many frames to go without running the bounding
 * box detector. Only relevant if maxFaces > 1. Defaults to 5.
 *  - `detectionConfidence` Threshold for discarding a prediction. Defaults to
 * 0.9.
 *  - `maxFaces` The maximum number of faces detected in the input. Should be
 * set to the minimum number for performance. Defaults to 10.
 *  - `iouThreshold` A float representing the threshold for deciding whether
 * boxes overlap too much in non-maximum suppression. Must be between [0, 1].
 * Defaults to 0.3.
 *  - `scoreThreshold` A threshold for deciding when to remove boxes based
 * on score in non-maximum suppression. Defaults to 0.75.
 */
export async function load({
  maxContinuousChecks = 5,
  detectionConfidence = 0.9,
  maxFaces = 10,
  iouThreshold = 0.3,
  scoreThreshold = 0.75
} = {}): Promise<FaceMesh> {
  const [blazeFace, blazeMeshModel] = await Promise.all([
    loadDetectorModel(maxFaces, iouThreshold, scoreThreshold), loadMeshModel()
  ]);

  const faceMesh = new FaceMesh(
      blazeFace, blazeMeshModel, maxContinuousChecks, detectionConfidence,
      maxFaces);
  return faceMesh;
}

async function loadDetectorModel(
    maxFaces: number, iouThreshold: number,
    scoreThreshold: number): Promise<blazeface.BlazeFaceModel> {
  return blazeface.load({maxFaces, iouThreshold, scoreThreshold});
}

async function loadMeshModel(): Promise<tfconv.GraphModel> {
  return tfconv.loadGraphModel(FACEMESH_GRAPHMODEL_PATH, {fromTFHub: true});
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

      return tf.tidy(() => {
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
    mesh: (face.mesh).map(coord => {
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

  constructor(
      blazeFace: blazeface.BlazeFaceModel, blazeMeshModel: tfconv.GraphModel,
      maxContinuousChecks: number, detectionConfidence: number,
      maxFaces: number) {
    this.pipeline = new Pipeline(
        blazeFace, blazeMeshModel, MESH_MODEL_INPUT_WIDTH,
        MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, maxFaces);

    this.detectionConfidence = detectionConfidence;
  }

  static getAnnotations(): {[key: string]: number[]} {
    return MESH_ANNOTATIONS;
  }

  /**
   * Returns an array of UV coordinates for the 468 facial keypoint vertices in
   * mesh_map.jpg. Can be used to map textures to the facial mesh.
   */
  static getUVCoords(): Array<[number, number]> {
    return UV_COORDS;
  }

  /**
   * Returns an array of faces in an image.
   *
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param returnTensors (defaults to `false`) Whether to return tensors as
   * opposed to values.
   * @param flipHorizontal Whether to flip/mirror the facial keypoints
   * horizontally. Should be true for videos that are flipped by default (e.g.
   * webcams).
   *
   * @return An array of AnnotatedPrediction objects.
   */
  async estimateFaces(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      returnTensors = false,
      flipHorizontal = false): Promise<AnnotatedPrediction[]> {
    const [, width] = getInputTensorDimensions(input);

    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });

    // Currently tfjs-core does not pack depthwiseConv because it fails for
    // very large inputs (https://github.com/tensorflow/tfjs/issues/1652).
    // TODO(annxingyuan): call tf.enablePackedDepthwiseConv when available
    // (https://github.com/tensorflow/tfjs/issues/2821)
    const savedWebglPackDepthwiseConvFlag =
        tf.env().get('WEBGL_PACK_DEPTHWISECONV');
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
    const predictions = await this.pipeline.predict(image);
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);

    image.dispose();

    if (predictions != null && predictions.length > 0) {
      return Promise.all(predictions.map(async (prediction: Prediction, i) => {
        const {coords, scaledCoords, box, flag} = prediction;
        let tensorsToRead: tf.Tensor[] = [flag];
        if (!returnTensors) {
          tensorsToRead = tensorsToRead.concat(
              [coords, scaledCoords, box.startPoint, box.endPoint]);
        }

        const tensorValues = await Promise.all(
            tensorsToRead.map(async (d: tf.Tensor) => d.array()));
        const flagValue = tensorValues[0] as number;

        flag.dispose();
        if (flagValue < this.detectionConfidence) {
          this.pipeline.clearRegionOfInterest(i);
        }

        if (returnTensors) {
          const annotatedPrediction: AnnotatedPrediction = {
            faceInViewConfidence: flagValue,
            mesh: coords,
            scaledMesh: scaledCoords,
            boundingBox: {
              // tslint:disable-next-line: no-unnecessary-type-assertion
              topLeft: box.startPoint.squeeze() as tf.Tensor1D,
              // tslint:disable-next-line: no-unnecessary-type-assertion
              bottomRight: box.endPoint.squeeze() as tf.Tensor1D
            }
          };

          if (flipHorizontal) {
            return flipFaceHorizontal(annotatedPrediction, width);
          }

          return annotatedPrediction;
        }

        const [coordsArr, coordsArrScaled, topLeft, bottomRight] =
                      tensorValues.slice(1) as [
                        Array<[number, number, number]>,
                        Array<[number, number, number]>,
                        [number, number],
                        [number, number]];

        scaledCoords.dispose();
        coords.dispose();

        let annotatedPrediction: AnnotatedPredictionValues = {
          faceInViewConfidence: flagValue,
          boundingBox: {topLeft, bottomRight},
          mesh: coordsArr,
          scaledMesh: coordsArrScaled
        };

        if (flipHorizontal) {
          annotatedPrediction =
              flipFaceHorizontal(annotatedPrediction, width) as
              AnnotatedPredictionValues;
        }

        const annotations:
            {[key: string]: Array<[number, number, number]>} = {};
        for (const key in MESH_ANNOTATIONS) {
          annotations[key] = MESH_ANNOTATIONS[key].map(
              index => annotatedPrediction.scaledMesh[index]);
        }
        annotatedPrediction['annotations'] = annotations;

        return annotatedPrediction;
      }));
    }

    return [];
  }
}
