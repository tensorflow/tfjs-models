/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

const MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/savedmodel/qna';

export {version} from './version';

/**
 * QuestionAndAnswer model loading is configurable using the following config
 * dictionary.
 *
 * `modelUrl`: An optional string that specifies custom url of the model. This
 * is useful for area/countries that don't have access to the model hosted on
 * GCP.
 */
export interface ModelConfig {
  modelUrl?: string;
}

export interface DetectedAnswer {
  answer: string;
  startIndex: number;
  endIndex: number;
  score: number;
}

export async function load(config: ModelConfig = {}) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  const modelUrl = config.modelUrl;

  const qna = new QuestionAndAnswer(modelUrl);
  await qna.load();
  return qna;
}

export const MAX_ANS_LEN = 32;
export const MAX_QUERY_LEN = 64;
export const MAX_SEQ_LEN = 384;
export const PREDICT_ANS_NUM = 5;
export const OUTPUT_OFFSET = 1;

export class QuestionAndAnswer {
  private modelPath: string;
  private model: tfconv.GraphModel;

  constructor(modelUrl?: string) {
    this.modelPath = modelUrl || `${MODEL_PATH}/model.json`;
  }

  async load() {
    this.model = await tfconv.loadGraphModel(this.modelPath);
    const inputIds = tf.ones([1, MAX_SEQ_LEN], 'int32');
    const segmentIds = tf.ones([1, MAX_SEQ_LEN], 'int32');
    const inputMask = tf.ones([1, MAX_SEQ_LEN], 'int32');
    const globalStep = tf.scalar(1, 'int32');
    // Warmup the model.
    const result = await this.model.executeAsync({
      input_ids: inputIds,
      segment_ids: segmentIds,
      input_mask: inputMask,
      global_step: globalStep
    }) as tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
  }

  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   */
  private async infer(question: string, passage: string):
      Promise<DetectedAnswer[]> {
    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0);
    });
    const height = batched.shape[1];
    const width = batched.shape[2];

    // model returns two tensors:
    // 1. box classification score with shape of [1, 1917, 90]
    // 2. box location with shape of [1, 1917, 1, 4]
    // where 1917 is the number of box detectors, 90 is the number of classes.
    // and 4 is the four coordinates of the box.
    const result = await this.model.executeAsync(batched) as tf.Tensor[];

    const scores = result[0].dataSync() as Float32Array;
    const boxes = result[1].dataSync() as Float32Array;

    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);

    const [maxScores, classes] =
        this.calculateMaxScores(scores, result[0].shape[1], result[0].shape[2]);

    const prevBackend = tf.getBackend();
    // run post process in cpu
    tf.setBackend('cpu');
    const indexTensor = tf.tidy(() => {
      const boxes2 =
          tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]]);
      return tf.image.nonMaxSuppression(
          boxes2, maxScores, maxNumBoxes, 0.5, 0.5);
    });

    const indexes = indexTensor.dataSync() as Float32Array;
    indexTensor.dispose();

    // restore previous backend
    tf.setBackend(prevBackend);

    return this.buildDetectedObjects(
        width, height, boxes, maxScores, indexes, classes);
  }

  /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   *
   */
  async detect(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxNumBoxes = 20): Promise<DetectedObject[]> {
    return this.infer(img, maxNumBoxes);
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
