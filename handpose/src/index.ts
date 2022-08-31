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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {HandDetector} from './hand';
import {MESH_ANNOTATIONS} from './keypoints';
import {Coords3D, HandPipeline, Prediction} from './pipeline';

// Load the bounding box detector model.
async function loadHandDetectorModel() {
  const HANDDETECT_MODEL_PATH =
      'https://tfhub.dev/mediapipe/tfjs-model/handdetector/1/default/1';
  return tfconv.loadGraphModel(HANDDETECT_MODEL_PATH, {fromTFHub: true});
}

const MESH_MODEL_INPUT_WIDTH = 256;
const MESH_MODEL_INPUT_HEIGHT = 256;

// Load the mesh detector model.
async function loadHandPoseModel() {
  const HANDPOSE_MODEL_PATH =
      'https://tfhub.dev/mediapipe/tfjs-model/handskeleton/1/default/1';
  return tfconv.loadGraphModel(HANDPOSE_MODEL_PATH, {fromTFHub: true});
}

// In single shot detector pipelines, the output space is discretized into a set
// of bounding boxes, each of which is assigned a score during prediction. The
// anchors define the coordinates of these boxes.
async function loadAnchors() {
  return tf.util
      .fetch(
          'https://tfhub.dev/mediapipe/tfjs-model/handskeleton/1/default/1/anchors.json?tfjs-format=file')
      .then(d => d.json());
}

export interface AnnotatedPrediction extends Prediction {
  annotations: {[key: string]: Array<[number, number, number]>};
}

/**
 * Load handpose.
 *
 * @param config A configuration object with the following properties:
 * - `maxContinuousChecks` How many frames to go without running the bounding
 * box detector. Defaults to infinity. Set to a lower value if you want a safety
 * net in case the mesh detector produces consistently flawed predictions.
 * - `detectionConfidence` Threshold for discarding a prediction. Defaults to
 * 0.8.
 * - `iouThreshold` A float representing the threshold for deciding whether
 * boxes overlap too much in non-maximum suppression. Must be between [0, 1].
 * Defaults to 0.3.
 * - `scoreThreshold` A threshold for deciding when to remove boxes based
 * on score in non-maximum suppression. Defaults to 0.75.
 */
export async function load({
  maxContinuousChecks = Infinity,
  detectionConfidence = 0.8,
  iouThreshold = 0.3,
  scoreThreshold = 0.5
} = {}): Promise<HandPose> {
  const [ANCHORS, handDetectorModel, handPoseModel] = await Promise.all(
      [loadAnchors(), loadHandDetectorModel(), loadHandPoseModel()]);

  const detector = new HandDetector(
      handDetectorModel, MESH_MODEL_INPUT_WIDTH, MESH_MODEL_INPUT_HEIGHT,
      ANCHORS, iouThreshold, scoreThreshold);
  const pipeline = new HandPipeline(
      detector, handPoseModel, MESH_MODEL_INPUT_WIDTH, MESH_MODEL_INPUT_HEIGHT,
      maxContinuousChecks, detectionConfidence);
  const handpose = new HandPose(pipeline);

  return handpose;
}

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipHandHorizontal(prediction: Prediction, width: number): Prediction {
  const {handInViewConfidence, landmarks, boundingBox} = prediction;
  return {
    handInViewConfidence,
    landmarks: landmarks.map(
        (coord: [number, number, number]): [number, number, number] => {
          return [width - 1 - coord[0], coord[1], coord[2]];
        }),
    boundingBox: {
      topLeft: [width - 1 - boundingBox.topLeft[0], boundingBox.topLeft[1]],
      bottomRight: [
        width - 1 - boundingBox.bottomRight[0], boundingBox.bottomRight[1]
      ]
    }
  };
}

export class HandPose {
  constructor(private readonly pipeline: HandPipeline) {}

  static getAnnotations(): {[key: string]: number[]} {
    return MESH_ANNOTATIONS;
  }

  /**
   * Finds hands in the input image.
   *
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param flipHorizontal Whether to flip the hand keypoints horizontally.
   * Should be true for videos that are flipped by default (e.g. webcams).
   */
  async estimateHands(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      flipHorizontal = false): Promise<AnnotatedPrediction[]> {
    const [, width] = getInputTensorDimensions(input);

    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast(input, 'float32'));
    });

    const result = await this.pipeline.estimateHand(image);
    image.dispose();

    if (result === null) {
      return [];
    }

    let prediction = result;
    if (flipHorizontal === true) {
      prediction = flipHandHorizontal(result, width);
    }

    const annotations: {[key: string]: Coords3D} = {};
    for (const key of Object.keys(MESH_ANNOTATIONS)) {
      annotations[key] =
          MESH_ANNOTATIONS[key].map(index => prediction.landmarks[index]);
    }

    return [{
      handInViewConfidence: prediction.handInViewConfidence,
      boundingBox: prediction.boundingBox,
      landmarks: prediction.landmarks,
      annotations
    }];
  }
}
