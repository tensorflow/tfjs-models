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

type MobileNetVersion = 1;
type MobileNetMultiplier = 0.25|0.50|0.75|1.0;
type MobileNetResolution = 224|192|160|128;

export default async function mobilenet(
    version: MobileNetVersion = 1, multiplier: MobileNetMultiplier = 1.0,
    resolution: MobileNetResolution = 224) {
  if (version != 1) {
    throw new Error(
        `Currently only MobileNet V1 is supported. Got version ${version}.`);
  }
  if ([0.25, 0.50, 0.75, 1.0].indexOf(multiplier) === -1) {
    throw new Error(
        `MobileNet constructed with invalid multiplier ` +
        `${multiplier}. Valid multipliers are 0.25, 0.50, 0.75, and 1.0.`);
  }
  if (resolution != 224) {
    throw new Error(
        `Currently only MobileNet with resolution 224 is supported. Got ` +
        `resolution ${resolution}.`)
  }

  const mobilenet = new MobileNet(version, multiplier, resolution);
  await mobilenet.load();
  return mobilenet;
}

class MobileNet {
  private path: string;
  private model: tf.Model;

  private offset: tf.Scalar;

  constructor(
      version: MobileNetVersion, multiplier: MobileNetMultiplier,
      private resolution: MobileNetResolution) {
    const multiplierStr =
        ({0.25: '0.25', 0.50: '0.50', 0.75: '0.75', 1.0: '1.0'})[multiplier];
    this.path =
        `${BASE_PATH}mobilenet_v${version}_${multiplierStr}_${resolution}/` +
        `model.json`;
    this.offset = tf.scalar(127.5);
  }

  async load() {
    this.model = await tf.loadModel(this.path);
    // Warmup the model.
    tf.tidy(
        () => this.model.predict(
            tf.zeros([0, this.resolution, this.resolution, 3])));
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
    const logits = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.fromPixels(img);
      }

      // Normalize the image from [0, 255] to [-1, 1].
      const normalized =
          img.toFloat().sub(this.offset).div(this.offset) as tf.Tensor3D;

      // Resize the image to
      let resized = normalized;
      if (img.shape[0] != this.resolution || img.shape[1] != this.resolution) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
            normalized, [this.resolution, this.resolution], alignCorners);
      }

      // Reshape to a single-element batch so we can pass it to predict.
      const batched = resized.reshape([1, this.resolution, this.resolution, 3]);

      return this.model.predict(batched) as tf.Tensor;
    });

    const classes = await getTopKClasses(logits, topk);

    logits.dispose();

    return classes;
  }
}

async function getTopKClasses(
    logits, topK): Promise<Array<{className: string, probability: number}>> {
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
    })
  }
  return topClassesAndProbs;
}
