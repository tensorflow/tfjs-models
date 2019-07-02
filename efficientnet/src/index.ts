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
import {EfficientNetBaseModel, EfficientNetInput, EfficientNetOutput,} from './types';

export class EfficientNet {
  private base: EfficientNetBaseModel;
  private model: Promise<tf.GraphModel>;
  private modelPath: string;
  public constructor(base: EfficientNetBaseModel, isQuantized = true) {
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
    this.base = base;

    this.modelPath = `${config['BASE_PATH']}/${
        isQuantized ? 'quantized/' : ''}${base}/model.json`;

    this.model = tf.loadGraphModel(this.modelPath);
  }

  private static normalize(image: tf.Tensor3D) {
    return tf.tidy(() => {
      const [height, width] = image.shape;
      const imageData = image.arraySync() as number[][][];
      const meanRGB = config['MEAN_RGB'].map(depth => depth * 255);
      const stddevRGB = config['STDDEV_RGB'].map(depth => depth * 255);
      for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
        for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
          imageData[columnIndex][rowIndex] =
              imageData[columnIndex][rowIndex].map(
                  (depth, channel) =>
                      (depth - meanRGB[channel]) / stddevRGB[channel]);
        }
      }
      return tf.tensor3d(imageData);
    });
  }

  private static cropAndResize(
      base: EfficientNetBaseModel, input: EfficientNetInput): tf.Tensor3D {
    return tf.tidy(() => {
      const image: tf.Tensor3D =
          (input instanceof tf.Tensor ? input : tf.browser.fromPixels(input))
              .toFloat();

      const [height, width] = image.shape;

      const imageSize = config['CROP_SIZE'][base];
      const cropPadding = config['CROP_PADDING'];
      const paddedCenterCropSize = Math.round(
          Math.min(width, height) *
          ((1.0 * imageSize) / (imageSize + cropPadding)));
      const offsetHeight = Math.round((height - paddedCenterCropSize + 1) / 2);
      const offsetWidth = Math.round((width - paddedCenterCropSize + 1) / 2);
      const normalizedBox = [
        offsetHeight / height,
        offsetWidth / width,
        (paddedCenterCropSize + offsetHeight) / height,
        (paddedCenterCropSize + offsetWidth) / width,
      ];
      const processedImage: tf.Tensor3D =
          tf.image
              .cropAndResize(
                  image.expandDims(0), [normalizedBox], [0],
                  [imageSize, imageSize])
              .squeeze([0]);
      return processedImage;
    });
  }

  private getTopKClasses(logits: tf.Tensor1D, topK: number) {
    const imagenetClasses: {[key: number]: string} = config.IMAGENET_CLASSES;
    const values = logits.dataSync();

    const valuesAndIndices = [];
    for (let idx = 0; idx < values.length; idx++) {
      valuesAndIndices.push({value: values[idx], index: idx});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let idx = 0; idx < topK; idx++) {
      topkValues[idx] = valuesAndIndices[idx].value;
      topkIndices[idx] = valuesAndIndices[idx].index;
    }

    const topClassesAndProbs = [];
    for (let idx = 0; idx < topkIndices.length; idx++) {
      topClassesAndProbs.push({
        className: imagenetClasses[topkIndices[idx]],
        probability: topkValues[idx],
      });
    }
    return topClassesAndProbs;
  }

  public static preprocess(
      base: EfficientNetBaseModel, input: EfficientNetInput) {
    return tf.tidy(() => {
      return EfficientNet.normalize(EfficientNet.cropAndResize(base, input))
          .expandDims(0);
    });
  }

  public async predict(input: EfficientNetInput, topK: number):
      Promise<EfficientNetOutput> {
    const model = await this.model;

    return tf.tidy(() => {
      const processedInput = EfficientNet.preprocess(this.base, input);
      const logits = model.predict(processedInput) as tf.Tensor1D;
      const predictions = this.getTopKClasses(logits, topK);
      return predictions;
    });
  }

  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    (await this.model).dispose();
  }
}
