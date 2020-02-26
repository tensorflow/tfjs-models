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

async function loadHandSkeletonModel() {
  const HANDTRACK_MODEL_PATH =
      'https://tfhub.dev/tensorflow/tfjs-model/handskeleton/1/default/1';
  return tfconv.loadGraphModel(HANDTRACK_MODEL_PATH, {fromTFHub: true});
}

async function loadAnchors() {
  return tf.util
      .fetch(
          'https://storage.googleapis.com/learnjs-data/handtrack_staging/anchors.json')
      .then(d => d.json());
}

export async function load({
  meshWidth = 256,
  meshHeight = 256,
  maxContinuousChecks = 100,
  detectionConfidence = 0.8,
  iouThreshold = 0.3,
  scoreThreshold = 0.5
} = {}) {
  const [ANCHORS, handDetectorModel, handSkeletonModel] = await Promise.all(
      [loadAnchors(), loadHandDetectorModel(), loadHandSkeletonModel()]);

  const detector = new HandDetector(
      handDetectorModel, meshWidth, meshHeight, ANCHORS, iouThreshold,
      scoreThreshold);
  const pipeline = new HandPipeline(
      detector, handSkeletonModel, maxContinuousChecks, detectionConfidence);

  return pipeline;
}
