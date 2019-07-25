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

import {config} from './config';
import {minAreaRect} from './minAreaRect';
import {TextDetectionConfig, TextDetectionInput, TextDetectionOptions, TextDetectionOutput} from './types';
import {computeScalingFactors, detect, getURL, resize} from './utils';

export {computeScalingFactors, detect, getURL, minAreaRect};

export const load = async (modelConfig: TextDetectionConfig = {
  quantizationBytes: 1
}) => {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. ` +
        `If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if (modelConfig.quantizationBytes) {
    if ([1, 2, 4].indexOf(modelConfig.quantizationBytes) === -1) {
      throw new Error(`Only quantization to 1, 2 and 4 bytes is supported.`);
    }
  } else if (!modelConfig.modelUrl) {
    throw new Error(
        `TextDetection can be constructed either by passing` +
        `the weights URL or by specifying one of the degree of quantization, ` +
        `out of 1, 2 and 4.` +
        `Aborting, since neither has been provided.`);
  }
  const url = getURL(modelConfig.quantizationBytes);
  const graphModel = await tfconv.loadGraphModel(modelConfig.modelUrl || url);
  const textDetection = new TextDetection(graphModel);
  return textDetection;
};

export class TextDetection {
  readonly model: tfconv.GraphModel;
  public constructor(graphModel: tfconv.GraphModel) {
    this.model = graphModel;
  }

  public preprocess(input: TextDetectionInput, resizeLength?: number) {
    if (!resizeLength) {
      resizeLength = config['RESIZE_LENGTH'];
    }
    return tf.tidy(() => {
      return resize(input, resizeLength).expandDims(0);
    });
  }

  public async detect(
      input: TextDetectionInput, textDetectionOptions: TextDetectionOptions = {
        resizeLength: config['RESIZE_LENGTH'],
        minTextBoxArea: config['MIN_TEXTBOX_AREA'],
        minConfidence: config['MIN_CONFIDENCE'],
        processPoints: minAreaRect
      }): Promise<TextDetectionOutput> {
    textDetectionOptions = {
      resizeLength: config['RESIZE_LENGTH'],
      minTextBoxArea: config['MIN_TEXTBOX_AREA'],
      minConfidence: config['MIN_CONFIDENCE'],
      processPoints: minAreaRect,
      ...textDetectionOptions
    };
    const kernelScores = this.predict(input, textDetectionOptions.resizeLength);
    const sides = new Array<number>(2);
    if (input instanceof tf.Tensor) {
      const [height, width] = input.shape;
      sides[0] = height;
      sides[1] = width;
    } else {
      sides[0] = input.height;
      sides[1] = input.width;
    }
    const boxes =
        await detect(kernelScores, sides[0], sides[1], textDetectionOptions);
    tf.dispose(kernelScores);
    return boxes;
  }

  public predict(
      input: TextDetectionInput,
      resizeLength = config['RESIZE_LENGTH']): tf.Tensor3D {
    return tf.tidy(() => {
      const processedInput = this.preprocess(input, resizeLength);
      return (this.model.predict(processedInput) as tf.Tensor4D).squeeze([0]) as
          tf.Tensor3D;
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
