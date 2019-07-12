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

import * as tf from '@tensorflow/tfjs';

import {config} from './config';
import {EfficientNetBaseModel, EfficientNetInput, EfficientNetOutput, QuantizationBytes,} from './types';
import {cropAndResize, getTopKClasses, normalize} from './utils';

export async function load(
    base: EfficientNetBaseModel = 'b0',
    quantizationBytes: QuantizationBytes = 2) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. ` +
        `If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  if (['b0', 'b1', 'b2', 'b3', 'b4', 'b5'].indexOf(base) === -1) {
    throw new Error(
        `EfficientNet cannot be constructed ` +
        `with an invalid base model ${base}. ` +
        `Try one of 'b0', 'b1', 'b2', 'b3', 'b4' or 'b5'.`);
  }
  if ([1, 2, 4].indexOf(quantizationBytes) === -1) {
    throw new Error(`Only quantization to 1, 2 or 4 bytes is supported.`);
  }

  const efficientnet = new EfficientNet(base);
  await efficientnet.load();
  return efficientnet;
}
export class EfficientNet {
  private base: EfficientNetBaseModel;
  private model: tf.GraphModel;
  private modelPath: string;
  public constructor(
      base: EfficientNetBaseModel, quantizationBytes: QuantizationBytes = 2) {
    this.base = base;

    this.modelPath = `${config['BASE_PATH']}/${
        ([1, 2].indexOf(quantizationBytes) !== -1) ?
            `quantized/${quantizationBytes}/` :
            ''}${base}/model.json`;
  }

  public hasLoaded() {
    return !!this.model;
  }

  public async load() {
    this.model = await tf.loadGraphModel(this.modelPath);
    // Warm the model up.
    const processedInput = this.preprocess(tf.zeros([227, 227, 3]));
    const result = await this.model.predict(processedInput) as tf.Tensor1D;
    await result.data();
    result.dispose();
  }

  public preprocess(input: EfficientNetInput) {
    return tf.tidy(() => {
      return normalize(cropAndResize(this.base, input)).expandDims(0);
    });
  }

  public async predict(input: EfficientNetInput, topK: number):
      Promise<EfficientNetOutput> {
    return tf.tidy(() => {
      const processedInput = this.preprocess(input);
      const logits = this.model.predict(processedInput) as tf.Tensor1D;
      const predictions = getTopKClasses(logits, topK);
      return predictions;
    });
  }

  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
