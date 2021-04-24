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
import {Coord2D, Coords3D} from './util';
import {UV_COORDS} from './uv_coords';

const FACEMESH_GRAPHMODEL_PATH =
    'https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1';
const IRIS_GRAPHMODEL_PATH =
    'https://tfhub.dev/mediapipe/tfjs-model/iris/1/default/2';
const MESH_MODEL_INPUT_WIDTH = 192;
const MESH_MODEL_INPUT_HEIGHT = 192;

export interface EstimateFacesConfig {
  /**
   * The image to classify. Can be a tensor, DOM element image, video, or
   * canvas.
   */
  input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement;
  /** Whether to return tensors as opposed to values. */
  returnTensors?: boolean;
  /** Whether to flip/mirror the facial keypoints horizontally. */
  flipHorizontal?: boolean;
  /**
   * Whether to return keypoints for the irises. Disabling may improve
   * performance. Defaults to true.
   */
  predictIrises?: boolean;
}

const PREDICTION_VALUES = 'MediaPipePredictionValues';
type PredictionValuesKind = typeof PREDICTION_VALUES;

interface AnnotatedPredictionValues {
  kind: PredictionValuesKind;
  /** Probability of the face detection. */
  faceInViewConfidence: number;
  boundingBox: {
    /** The upper left-hand corner of the face. */
    topLeft: Coord2D,
    /** The lower right-hand corner of the face. */
    bottomRight: Coord2D
  };
  /** Facial landmark coordinates. */
  mesh: Coords3D;
  /** Facial landmark coordinates normalized to input dimensions. */
  scaledMesh: Coords3D;
  /** Annotated keypoints. */
  annotations?: {[key: string]: Coords3D};
}

const PREDICTION_TENSORS = 'MediaPipePredictionTensors';
type PredictionTensorsKind = typeof PREDICTION_TENSORS;

interface AnnotatedPredictionTensors {
  kind: PredictionTensorsKind;
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
 *  - `shouldLoadIrisModel` Whether to also load the iris detection model.
 * Defaults to true.
 *  - `modelUrl` Optional param for specifying a custom facemesh model url or
 * a `tf.io.IOHandler` object.
 *  - `detectorModelUrl` Optional param for specifying a custom blazeface model
 * url or a `tf.io.IOHandler` object.
 *  - `irisModelUrl` Optional param for specifying a custom iris model url or
 * a `tf.io.IOHandler` object.
 */
export async function load(config: {
  maxContinuousChecks?: number,
  detectionConfidence?: number,
  maxFaces?: number,
  iouThreshold?: number,
  scoreThreshold?: number,
  shouldLoadIrisModel?: boolean,
  modelUrl?: string|tf.io.IOHandler,
  detectorModelUrl?: string|tf.io.IOHandler,
  irisModelUrl?: string|tf.io.IOHandler,
}): Promise<FaceMesh> {
  const {
    maxContinuousChecks = 5,
    detectionConfidence = 0.9,
    maxFaces = 10,
    iouThreshold = 0.3,
    scoreThreshold = 0.75,
    shouldLoadIrisModel = true,
    modelUrl,
    detectorModelUrl,
    irisModelUrl
  } = config;

  let models;
  if (shouldLoadIrisModel) {
    models = await Promise.all([
      loadDetectorModel(
        detectorModelUrl, maxFaces, iouThreshold, scoreThreshold
      ),
      loadMeshModel(modelUrl),
      loadIrisModel(irisModelUrl)
    ]);
  } else {
    models = await Promise.all([
      loadDetectorModel(
        detectorModelUrl, maxFaces, iouThreshold, scoreThreshold
      ),
      loadMeshModel(modelUrl)
    ]);
  }

  const faceMesh = new FaceMesh(
      models[0], models[1], maxContinuousChecks, detectionConfidence, maxFaces,
      shouldLoadIrisModel ? models[2] : null);
  return faceMesh;
}

async function loadDetectorModel(
    modelUrl: string|tf.io.IOHandler,
    maxFaces: number,
    iouThreshold: number,
    scoreThreshold: number
): Promise<blazeface.BlazeFaceModel> {
  return blazeface.load({modelUrl, maxFaces, iouThreshold, scoreThreshold});
}

async function loadMeshModel(modelUrl?: string|
                             tf.io.IOHandler): Promise<tfconv.GraphModel> {
  if (modelUrl != null) {
    return tfconv.loadGraphModel(modelUrl);
  }
  return tfconv.loadGraphModel(FACEMESH_GRAPHMODEL_PATH, {fromTFHub: true});
}

async function loadIrisModel(modelUrl?: string|
                             tf.io.IOHandler): Promise<tfconv.GraphModel> {
  if (modelUrl != null) {
    return tfconv.loadGraphModel(modelUrl);
  }
  return tfconv.loadGraphModel(IRIS_GRAPHMODEL_PATH, {fromTFHub: true});
}

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|HTMLCanvasElement): Coord2D {
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
                tf.slice((face.boundingBox.topLeft as tf.Tensor1D), 0, 1)),
            tf.slice((face.boundingBox.topLeft as tf.Tensor1D), 1, 1)
          ]),
          tf.concat([
            tf.sub(
                imageWidth - 1,
                tf.slice((face.boundingBox.bottomRight as tf.Tensor1D), 0, 1)),
            tf.slice((face.boundingBox.bottomRight as tf.Tensor1D), 1, 1)
          ]),
          tf.mul(tf.sub(subtractBasis, face.mesh), multiplyBasis),
          tf.mul(tf.sub(subtractBasis, face.scaledMesh), multiplyBasis)
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
    scaledMesh: (face.scaledMesh as Coords3D).map(coord => {
      const flippedCoord = coord.slice(0);
      flippedCoord[0] = imageWidth - 1 - coord[0];
      return flippedCoord;
    })
  });
}

export interface MediaPipeFaceMesh {
  kind: 'MediaPipeFaceMesh';
  estimateFaces(config: EstimateFacesConfig): Promise<AnnotatedPrediction[]>;
}

class FaceMesh implements MediaPipeFaceMesh {
  private pipeline: Pipeline;
  private detectionConfidence: number;

  public kind = 'MediaPipeFaceMesh' as const ;

  constructor(
      blazeFace: blazeface.BlazeFaceModel, blazeMeshModel: tfconv.GraphModel,
      maxContinuousChecks: number, detectionConfidence: number,
      maxFaces: number, irisModel: tfconv.GraphModel|null) {
    this.pipeline = new Pipeline(
        blazeFace, blazeMeshModel, MESH_MODEL_INPUT_WIDTH,
        MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, maxFaces, irisModel);

    this.detectionConfidence = detectionConfidence;
  }

  static getAnnotations(): {[key: string]: number[]} {
    return MESH_ANNOTATIONS;
  }

  /**
   * Returns an array of UV coordinates for the 468 facial keypoint vertices in
   * mesh_map.jpg. Can be used to map textures to the facial mesh.
   */
  static getUVCoords(): Coord2D[] {
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
   * @param predictIrises
   *
   * @return An array of AnnotatedPrediction objects.
   */
  async estimateFaces(config: EstimateFacesConfig):
      Promise<AnnotatedPrediction[]> {
    const {
      returnTensors = false,
      flipHorizontal = false,
      predictIrises = true
    } = config;
    let input = config.input;

    if (predictIrises && this.pipeline.irisModel == null) {
      throw new Error(
          'The iris model was not loaded as part of facemesh. ' +
          'Please initialize the model with ' +
          'facemesh.load({shouldLoadIrisModel: true}).');
    }

    const [, width] = getInputTensorDimensions(input);

    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast((input as tf.Tensor), 'float32'), 0);
    });

    let predictions;
    if (tf.getBackend() === 'webgl') {
      // Currently tfjs-core does not pack depthwiseConv because it fails for
      // very large inputs (https://github.com/tensorflow/tfjs/issues/1652).
      // TODO(annxingyuan): call tf.enablePackedDepthwiseConv when available
      // (https://github.com/tensorflow/tfjs/issues/2821)
      const savedWebglPackDepthwiseConvFlag =
          tf.env().get('WEBGL_PACK_DEPTHWISECONV');
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
      predictions = await this.pipeline.predict(image, predictIrises);
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    } else {
      predictions = await this.pipeline.predict(image, predictIrises);
    }

    image.dispose();

    if (predictions != null && predictions.length > 0) {
      return Promise.all(predictions.map(async (prediction: Prediction, i) => {
        const {coords, scaledCoords, box, flag} = prediction;
        let tensorsToRead: tf.Tensor[] = [flag];
        if (!returnTensors) {
          tensorsToRead = tensorsToRead.concat([coords, scaledCoords]);
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
            kind: PREDICTION_TENSORS,
            faceInViewConfidence: flagValue,
            mesh: coords,
            scaledMesh: scaledCoords,
            boundingBox: {
              topLeft: tf.tensor1d(box.startPoint),
              bottomRight: tf.tensor1d(box.endPoint)
            }
          };

          if (flipHorizontal) {
            return flipFaceHorizontal(annotatedPrediction, width);
          }

          return annotatedPrediction;
        }

        const [coordsArr, coordsArrScaled] =
            tensorValues.slice(1) as [Coords3D, Coords3D];

        scaledCoords.dispose();
        coords.dispose();

        let annotatedPrediction: AnnotatedPredictionValues = {
          kind: PREDICTION_VALUES,
          faceInViewConfidence: flagValue,
          boundingBox: {topLeft: box.startPoint, bottomRight: box.endPoint},
          mesh: coordsArr,
          scaledMesh: coordsArrScaled
        };

        if (flipHorizontal) {
          annotatedPrediction =
              flipFaceHorizontal(annotatedPrediction, width) as
              AnnotatedPredictionValues;
        }

        const annotations: {[key: string]: Coords3D} = {};
        for (const key in MESH_ANNOTATIONS) {
          if (predictIrises || key.includes('Iris') === false) {
            annotations[key] = MESH_ANNOTATIONS[key].map(
                index => annotatedPrediction.scaledMesh[index]);
          }
        }
        annotatedPrediction['annotations'] = annotations;

        return annotatedPrediction;
      }));
    }

    return [];
  }
}
