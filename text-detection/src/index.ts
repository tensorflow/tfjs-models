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
import {TextDetectionInput, TextDetectionOutput} from './types';
export class TextDetection {
  private model: Promise<tf.GraphModel>;
  private modelPath: string;
  private base = 'psenet';
  public constructor(quantizationBytes?: number) {
    if (tf == null) {
      throw new Error(
          `Cannot find TensorFlow.js. ` +
          `If you are using a <script> tag, please ` +
          `also include @tensorflow/tfjs on the page before using this model.`);
    }

    if (quantizationBytes && [1, 2].indexOf(quantizationBytes) === -1) {
      throw new Error(`Only quantization to 1 or 2 bytes is supported.`);
    }
    this.modelPath = `${config['BASE_PATH']}/${
        quantizationBytes ? `quantized/${quantizationBytes}/` :
                            ''}${this.base}/model.json`;

    this.model = tf.loadGraphModel(this.modelPath);
  }

  private static computeResizeRatios(height: number, width: number):
      [number, number] {
    const maxSide = Math.max(width, height)
    const ratio = maxSide > config['MAX_SIDE_LENGTH'] ?
        config['MAX_SIDE_LENGTH'] / maxSide :
        1;

    const resizeRatios = [height, width].map((side: number) => {
      const roundedSide = Math.round(side * ratio);
      return (roundedSide % 32 === 0 ?
                  roundedSide :
                  (Math.floor(roundedSide / 32) + 1) * 32) /
          side;
    }) as [number, number];
    return resizeRatios;
  }

  private static cropAndResize(input: TextDetectionInput): tf.Tensor3D {
    return tf.tidy(() => {
      const image: tf.Tensor3D =
          (input instanceof tf.Tensor ? input : tf.browser.fromPixels(input))
              .toFloat();

      const [height, width] = image.shape;
      const resizeRatios = TextDetection.computeResizeRatios(height, width);
      const processedImage =
          tf.image.resizeBilinear(image, [height, width].map((side, idx) => {
            return Math.round(side * resizeRatios[idx]);
          }) as [number, number]);

      return processedImage;
    });
  }

  public static preprocess(input: TextDetectionInput) {
    return tf.tidy(() => {
      return TextDetection.cropAndResize(input).expandDims(0);
    });
  }

  public async predict(input: TextDetectionInput):
      Promise<TextDetectionOutput> {
    const model = await this.model;
    return tf.tidy(() => {
      const processedInput = TextDetection.preprocess(input);
      const predictions =
          (model.predict(processedInput) as tf.Tensor4D).squeeze([0]) as
          tf.Tensor2D;
      console.log(predictions.shape);
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
