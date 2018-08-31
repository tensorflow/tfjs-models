/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import {CLASSES} from './classes';

const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/';

export type ObjectDetectionBaseModel =
    'ssd_mobilenet_v1'|'ssd_mobilenet_v2'|'ssdlite_mobilenet_v2';

export interface DetectedObject {
  bbox: [number, number, number, number];  // [x, y, width, height]
  class: string;
  score: number;
}

export async function load(
    base: ObjectDetectionBaseModel = 'ssdlite_mobilenet_v2') {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  if (['ssd_mobilenet_v1', 'ssd_mobilenet_v2', 'ssdlite_mobilenet_v2'].indexOf(
          base) === -1) {
    throw new Error(
        `ObjectDetection constructed with invalid base model ` +
        `${base}. Valid multipliers are 'ssd_mobilenet_v1',` +
        ` 'ssd_mobilenet_v2' and 'ssdlite_mobilenet_v2'.`);
  }

  const objectDetection = new ObjectDetection(base);
  await objectDetection.load();
  return objectDetection;
}

export class ObjectDetection {
  public endpoints: string[];

  private modelPath: string;
  private weightPath: string;
  private model: tf.FrozenModel;

  constructor(base: ObjectDetectionBaseModel) {
    this.modelPath = `${BASE_PATH}${base}/` +
        `tensorflowjs_model.pb`;
    this.weightPath = `${BASE_PATH}${base}/` +
        `weights_manifest.json`;
  }

  async load() {
    this.model = await tf.loadFrozenModel(this.modelPath, this.weightPath);

    // Warmup the model.
    const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as
        tf.Tensor[];
    result.map(async (t) => await t.data());
    result.map(async (t) => t.dispose());
  }

  /**
   * Infers through the model. Optionally takes an endpoint to return an
   * intermediate activation.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxDetectionSize The max count of detected objects.
   */
  private async infer(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxDetectionSize: number): Promise<DetectedObject[]> {
    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.fromPixels(img);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.reshape([1, ...img.shape]);
    })
    const width = batched.shape[2];
    const height = batched.shape[1];
    const result = await this.model.executeAsync(batched) as tf.Tensor[];

    const scores = result[0].dataSync() as Float32Array;
    const boxes = result[1].dataSync() as Float32Array;

    // clean the tensors
    batched.dispose();
    result[0].dispose();
    result[1].dispose();

    const [maxScores, classes] =
        this.calculateMaxScores(scores, result[0].shape);
    tf.setBackend('cpu');
    const boxes2 = tf.tensor2d(boxes, [1917, 4]);
    const indexes =
        tf.image
            .nonMaxSuppression(boxes2, maxScores, maxDetectionSize, 0.5, 0.5)
            .dataSync() as Float32Array;
    tf.setBackend('webgl');

    return this.buildDetectedObjects(
        width, height, boxes, maxScores, indexes, classes);
  }

  private buildDetectedObjects(
      width: number, height: number, boxes: Float32Array, scores: number[],
      indexes: Float32Array, classes: number[]): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox as [number, number, number, number],
        class: CLASSES[classes[indexes[i]] + 1].displayName,
        score: scores[indexes[i]]
      });
    }
    return objects;
  }

  private calculateMaxScores(scores: Float32Array, shape: number[]):
      [number[], number[]] {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < shape[1]; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < shape[2]; j++) {
        if (scores[i * shape[2] + j] > max) {
          max = scores[i * shape[2] + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   * @param maxDetectionSize The max count of detected objects, default to 20.
   *
   */
  async detect(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxDetectionSize = 20): Promise<DetectedObject[]> {
    return this.infer(img, maxDetectionSize);
  }

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
