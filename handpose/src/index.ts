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
import {HandPipeline} from './pipeline';

async function loadHandDetectorModel() {
  const HANDDETECT_MODEL_PATH =
      'https://tfhub.dev/tensorflow/tfjs-model/handdetector/1/default/1';

  return tfconv.loadGraphModel(HANDDETECT_MODEL_PATH, {fromTFHub: true});
}

const MESH_MODEL_INPUT_WIDTH = 256;
const MESH_MODEL_INPUT_HEIGHT = 256;

async function loadHandMeshModel() {
  const HANDPOSE_MODEL_PATH =
      'https://tfhub.dev/tensorflow/tfjs-model/handskeleton/1/default/1';
  return tfconv.loadGraphModel(HANDPOSE_MODEL_PATH, {fromTFHub: true});
}

// In single shot detector pipelines, the output space is discretized into a set
// of bounding boxes, each of which is assigned a score during prediction. The
// anchors define the coordinates of these boxes.
async function loadAnchors() {
  return tf.util
      .fetch(
          'https://storage.googleapis.com/learnjs-data/handtrack_staging/anchors.json')  // TODO: Where should these assets live?
      .then(d => d.json());
}

export async function load({
  maxContinuousChecks = Infinity,
  detectionConfidence = 0.8,
  iouThreshold = 0.3,
  scoreThreshold = 0.5
} = {}) {
  const [ANCHORS, handDetectorModel, handMeshModel] = await Promise.all(
      [loadAnchors(), loadHandDetectorModel(), loadHandMeshModel()]);

  const detector = new HandDetector(
      handDetectorModel, MESH_MODEL_INPUT_WIDTH, MESH_MODEL_INPUT_HEIGHT,
      ANCHORS, iouThreshold, scoreThreshold);
  const pipeline = new HandPipeline(
      detector, handMeshModel, maxContinuousChecks, detectionConfidence);

  return pipeline;
}
