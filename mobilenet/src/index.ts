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
import {IMAGENET_CLASSES} from './imagenet_classes';

const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/';
const IMAGE_SIZE = 224;

export type MobileNetVersion = 1;
export type MobileNetAlpha = 0.25|0.50|0.75|1.0;

export async function load(
    version: MobileNetVersion = 1, alpha: MobileNetAlpha = 1.0) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if (version !== 1) {
    throw new Error(
        `Currently only MobileNet V1 is supported. Got version ${version}.`);
  }
  if ([0.25, 0.50, 0.75, 1.0].indexOf(alpha) === -1) {
    throw new Error(
        `MobileNet constructed with invalid alpha ` +
        `${alpha}. Valid multipliers are 0.25, 0.50, 0.75, and 1.0.`);
  }

  const mobilenet = new MobileNet(version, alpha);
  await mobilenet.load();
  return mobilenet;
}

export class MobileNet {
  public endpoints: string[];

  private path: string;
  private model: tf.Model;
  private intermediateModels: {[layerName: string]: tf.Model} = {};

  private normalizationOffset: tf.Scalar;

  constructor(version: MobileNetVersion, alpha: MobileNetAlpha) {
    const multiplierStr =
        ({0.25: '0.25', 0.50: '0.50', 0.75: '0.75', 1.0: '1.0'})[alpha];
    this.path =
        `${BASE_PATH}mobilenet_v${version}_${multiplierStr}_${IMAGE_SIZE}/` +
        `model.json`;
    this.normalizationOffset = tf.scalar(127.5);
  }

  async load() {
    this.model = await tf.loadModel(this.path);
    this.endpoints = this.model.layers.map(l => l.name);

    // Warmup the model.
    const result = tf.tidy(
                       () => this.model.predict(tf.zeros(
                           [1, IMAGE_SIZE, IMAGE_SIZE, 3]))) as tf.Tensor;
    await result.data();
    result.dispose();
  }

  /**
   * Infers through the model. Optionally takes an endpoint to return an
   * intermediate activation.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param endpoint The endpoint to infer through. If not defined, returns
   * logits.
   */
  infer(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      endpoint?: string): tf.Tensor {
    if (endpoint != null && this.endpoints.indexOf(endpoint) === -1) {
      throw new Error(
          `Unknown endpoint ${endpoint}. Available endpoints: ` +
          `${this.endpoints}.`);
    }

    return tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.fromPixels(img);
      }

      // Normalize the image from [0, 255] to [-1, 1].
      const normalized = img.toFloat()
                             .sub(this.normalizationOffset)
                             .div(this.normalizationOffset) as tf.Tensor3D;

      // Resize the image to
      let resized = normalized;
      if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
            normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
      }

      // Reshape to a single-element batch so we can pass it to predict.
      const batched = resized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      let model: tf.Model;
      if (endpoint == null) {
        model = this.model;
      } else {
        if (this.intermediateModels[endpoint] == null) {
          const layer = this.model.layers.find(l => l.name === endpoint);
          this.intermediateModels[endpoint] =
              tf.model({inputs: this.model.inputs, outputs: layer.output});
        }
        model = this.intermediateModels[endpoint];
      }

      return model.predict(batched) as tf.Tensor2D;
    });
  }

  /**
   * Classifies an image from the 1000 ImageNet classes returning a map of
   * the most likely class names to their probability.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param topk How many top values to use. Defaults to 3.
   */
  async classify(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      topk = 3): Promise<Array<{className: string, probability: number}>> {
    const logits = this.infer(img) as tf.Tensor2D;

    const classes = await getTopKClasses(logits, topk);

    logits.dispose();

    return classes;
  }
}

async function getTopKClasses(logits: tf.Tensor2D, topK: number):
    Promise<Array<{className: string, probability: number}>> {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
}
