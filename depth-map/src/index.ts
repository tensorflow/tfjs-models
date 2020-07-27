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
import * as tfnode from '@tensorflow/tfjs-node';

const INPUT_SIZE = 224;

export interface FastDepth {
  load(): Promise<void>;
  predict(img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement): tf.Tensor;
}

export class DepthPredict implements FastDepth {
  private model: tfconv.GraphModel;
  private normalizationConstant: number;

  constructor(public modelUrl: string, public inputMin: number = 0, public inputMax: number = 255, public rawOutput: boolean = false) {
    this.normalizationConstant = (inputMax - inputMin);
  }

  public async load() {
    const handler = tfnode.io.fileSystem(this.modelUrl);
    this.model = await tfconv.loadGraphModel(handler);

  // Warmup the model.
    const result = tf.tidy(() => this.model.predict(tf.zeros([1, 3, INPUT_SIZE, INPUT_SIZE]))) as tf.Tensor;
    await result.data();
    result.dispose();
  }
  
  /**
   * Run depth inference through model
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   */
  public predict(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement): tf.Tensor {
      return tf.tidy(() => {
        if(!(img instanceof tf.Tensor)) {
          img = tf.browser.fromPixels(img);
        }
        // Normalize input from [inputMin, inputMax] to [0,1]
        // Resize the image to
        let resized = img.toFloat();
        if (img.shape[0] !== INPUT_SIZE || img.shape[1] !== INPUT_SIZE) {
          const alignCorners = true;
          resized = tf.image.resizeBilinear(img, [INPUT_SIZE, INPUT_SIZE], alignCorners);
        }

        const reshaped = tf.transpose(resized, [2, 0, 1]); // change image from [224,224,3] to [3,224,224]
        const batched = reshaped.reshape([1, 3, INPUT_SIZE, INPUT_SIZE]);
        const normalized: tf.Tensor3D = batched.sub(this.inputMin).div(this.normalizationConstant);
        const out: tf.Tensor = (this.model.predict(normalized) as tf.Tensor);
        const resizeOut = out.reshape([1, INPUT_SIZE, INPUT_SIZE]);
        if (this.rawOutput) {
          return resizeOut;
        } else {
          const reshapedOut = tf.transpose(resizeOut, [1, 2, 0]); // change output from [1, 224, 224] to [224, 224, 1]
          return reshapedOut;
        }
      });
  }
}
