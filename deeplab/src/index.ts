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

import {DeepLabInput, DeepLabOutput, SemanticSegmentationBaseModel, SemanticSegmentationConfig} from './types';
import {getColormap, getLabels, getURL, toInputTensor, toSegmentationImage} from './utils';

export {getColormap, getLabels, getURL, toSegmentationImage};

export async function load(modelConfig: SemanticSegmentationConfig = {
  base: 'pascal',
  quantizationBytes: 2
}) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js.` +
        ` If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if ([1, 2, 4].indexOf(modelConfig.quantizationBytes) === -1) {
    throw new Error(`Only quantization to 1, 2 or 4 bytes is supported.`);
  }
  if (modelConfig.base) {
    if (['pascal', 'cityscapes', 'ade20k'].indexOf(modelConfig.base) === -1) {
      throw new Error(
          `SemanticSegmentation cannot be constructed ` +
          `with an invalid base model ${modelConfig.base}. ` +
          `Try one of 'pascal', 'cityscapes' and 'ade20k'.`);
    }
  } else if (!modelConfig.modelUrl) {
    throw new Error(
        `SemanticSegmentation can be constructed either by passing` +
        `the weights URL or one of the supported base model names from` +
        `'pascal', 'cityscapes' and 'ade20k'.` +
        `Aborting, since none has been provided.`);
  }
  const url = getURL(modelConfig.base, modelConfig.quantizationBytes);
  const graphModel = await tfconv.loadGraphModel(modelConfig.modelUrl || url);
  const semanticSegmentation =
      new SemanticSegmentation(graphModel, modelConfig.base);
  return semanticSegmentation;
}

export class SemanticSegmentation {
  readonly model: tfconv.GraphModel;
  readonly base: SemanticSegmentationBaseModel;
  constructor(
      graphModel: tfconv.GraphModel,
      base?: SemanticSegmentationBaseModel,
  ) {
    this.model = graphModel;
    this.base = base;
  }

  public predict(input: DeepLabInput): tf.Tensor2D {
    return tf.tidy(() => {
      const data = toInputTensor(input);
      return tf.squeeze(this.model.execute(data) as tf.Tensor);
    }) as tf.Tensor2D;
  }

  public async segment(
      input: DeepLabInput, canvas?: HTMLCanvasElement,
      colormap = getColormap(this.base),
      labels = getLabels(this.base)): Promise<DeepLabOutput> {
    const rawSegmentationMap = tf.tidy(() => this.predict(input));

    const [height, width] = rawSegmentationMap.shape;
    const {legend, segmentationMap} =
        await toSegmentationImage(colormap, labels, rawSegmentationMap, canvas);

    tf.dispose(rawSegmentationMap);

    return {legend, height, width, segmentationMap};
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
