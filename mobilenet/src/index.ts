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

import {IMAGENET_CLASSES} from './imagenet_classes';
export {version} from './version';

const IMAGE_SIZE = 224;

/** @docinline */
export type MobileNetVersion = 1|2;
/** @docinline */
export type MobileNetAlpha = 0.25|0.50|0.75|1.0;

/**
 * Mobilenet model loading configuration
 *
 * Users should provide a version and alpha *OR* a modelURL and inputRange.
 */
export interface ModelConfig {
  /**
   * The MobileNet version number. Use 1 for MobileNetV1, and 2 for
   * MobileNetV2. Defaults to 1.
   */
  version: MobileNetVersion;
  /**
   * Controls the width of the network, trading accuracy for performance. A
   * smaller alpha decreases accuracy and increases performance. Defaults
   * to 1.0.
   */
  alpha?: MobileNetAlpha;
  /**
   * Optional param for specifying the custom model url or an `tf.io.IOHandler`
   * object.
   */
  modelUrl?: string|tf.io.IOHandler;
  /**
   * The input range expected by the trained model hosted at the modelUrl. This
   * is typically [0, 1] or [-1, 1].
   */
  inputRange?: [number, number];
}

const EMBEDDING_NODES: {[version: string]: string} = {
  '1.00': 'module_apply_default/MobilenetV1/Logits/global_pool',
  '2.00': 'module_apply_default/MobilenetV2/Logits/AvgPool'
};

export interface MobileNetInfo {
  // Where to find the TFHub version of this model.
  url: string;
  // The expected limits of the color channel values, in [min, max] format.
  inputRange: [number, number];
}

const MODEL_INFO: {[version: string]: {[alpha: string]: MobileNetInfo}} = {
  '1.00': {
    '0.25': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/1',
      inputRange: [0, 1]
    },
    '0.50': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/1',
      inputRange: [0, 1]
    },
    '0.75': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/1',
      inputRange: [0, 1]
    },
    '1.00': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1',
      inputRange: [0, 1]
    }
  },
  '2.00': {
    '0.50': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/2',
      inputRange: [0, 1]
    },
    '0.75': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/2',
      inputRange: [0, 1]
    },
    '1.00': {
      url:
          'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
      inputRange: [0, 1]
    }
  }
};

// See ModelConfig documentation for expectations of provided fields.
export async function load(modelConfig: ModelConfig = {
  version: 1,
  alpha: 1.0
}): Promise<MobileNet> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  const versionStr = modelConfig.version.toFixed(2);
  const alphaStr = modelConfig.alpha ? modelConfig.alpha.toFixed(2) : '';
  let inputMin = -1;
  let inputMax = 1;
  // User provides versionStr / alphaStr.
  if (modelConfig.modelUrl == null) {
    if (!(versionStr in MODEL_INFO)) {
      throw new Error(
          `Invalid version of MobileNet. Valid versions are: ` +
          `${Object.keys(MODEL_INFO)}`);
    }
    if (!(alphaStr in MODEL_INFO[versionStr])) {
      throw new Error(
          `MobileNet constructed with invalid alpha ${
              modelConfig.alpha}. Valid ` +
          `multipliers for this version are: ` +
          `${Object.keys(MODEL_INFO[versionStr])}.`);
    }
    [inputMin, inputMax] = MODEL_INFO[versionStr][alphaStr].inputRange;
  }
  // User provides modelUrl & optional<inputRange>.
  if (modelConfig.inputRange != null) {
    [inputMin, inputMax] = modelConfig.inputRange;
  }
  const mobilenet = new MobileNetImpl(
      versionStr, alphaStr, modelConfig.modelUrl, inputMin, inputMax);
  await mobilenet.load();
  return mobilenet;
}

export interface MobileNet {
  load(): Promise<void>;
  infer(
      img: tf.Tensor|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      embedding?: boolean): tf.Tensor;
  classify(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      topk?: number): Promise<Array<{className: string, probability: number}>>;
}

class MobileNetImpl implements MobileNet {
  model: tfconv.GraphModel;

  // Values read from images are in the range [0.0, 255.0], but they must
  // be normalized to [min, max] before passing to the mobilenet classifier.
  // Different implementations of mobilenet have different values of [min, max].
  // We store the appropriate normalization parameters using these two scalars
  // such that:
  // out = (in / 255.0) * (inputMax - inputMin) + inputMin;
  private normalizationConstant: number;

  constructor(
      public version: string, public alpha: string,
      public modelUrl: string|tf.io.IOHandler, public inputMin = -1,
      public inputMax = 1) {
    this.normalizationConstant = (inputMax - inputMin) / 255.0;
  }

  async load() {
    if (this.modelUrl) {
      this.model = await tfconv.loadGraphModel(this.modelUrl);
      // Expect that models loaded by URL should be normalized to [-1, 1]
    } else {
      const url = MODEL_INFO[this.version][this.alpha].url;
      this.model = await tfconv.loadGraphModel(url, {fromTFHub: true});
    }

    // Warmup the model.
    const result = tf.tidy(
                       () => this.model.predict(tf.zeros(
                           [1, IMAGE_SIZE, IMAGE_SIZE, 3]))) as tf.Tensor;
    await result.data();
    result.dispose();
  }

  /**
   * Computes the logits (or the embedding) for the provided image.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   *     video, or canvas.
   * @param embedding If true, it returns the embedding. Otherwise it returns
   *     the 1000-dim logits.
   */
  infer(
      img: tf.Tensor|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      embedding = false): tf.Tensor {
    return tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }

      // Normalize the image from [0, 255] to [inputMin, inputMax].
      const normalized: tf.Tensor3D = tf.add(
          tf.mul(tf.cast(img, 'float32'), this.normalizationConstant),
          this.inputMin);

      // Resize the image to
      let resized = normalized;
      if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
            normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
      }

      // Reshape so we can pass it to predict.
      const batched = tf.reshape(resized, [-1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      let result: tf.Tensor2D;

      if (embedding) {
        const embeddingName = EMBEDDING_NODES[this.version];
        const internal =
            this.model.execute(batched, embeddingName) as tf.Tensor4D;
        result = tf.squeeze(internal, [1, 2]);
      } else {
        const logits1001 = this.model.predict(batched) as tf.Tensor2D;
        // Remove the very first logit (background noise).
        result = tf.slice(logits1001, [0, 1], [-1, 1000]);
      }

      return result;
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
  const softmax = tf.softmax(logits);
  const values = await softmax.data();
  softmax.dispose();

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
