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
import { config } from './config';
import {
  DeepLabInput,
  DeepLabOutput,
  Label,
  Legend,
  RawSegmentationMap,
  SegmentationData,
  SemanticSegmentationBaseModel,
} from './types';
import { getColormap, toInputTensor, translateLabels } from './utils';

export class SemanticSegmentation {
  private base: SemanticSegmentationBaseModel;
  private model: Promise<tf.GraphModel>;
  private modelPath: string;
  constructor(base: SemanticSegmentationBaseModel, isQuantized = false) {
    if (tf == null) {
      throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
          `also include @tensorflow/tfjs on the page before using this model.`
      );
    }
    if (['pascal', 'cityscapes', 'ade20k'].indexOf(base) === -1) {
      throw new Error(
        `SemanticSegmentation cannot be constructed ` +
          `with an invalid base model ${base}. ` +
          `Try one of 'pascal', 'cityscapes' and 'ade20k'.`
      );
    }
    this.base = base;

    this.modelPath = `${config['BASE_PATH']}/${
      isQuantized ? 'quantized/' : ''
    }${base}/model.json`;

    this.model = tf.loadGraphModel(this.modelPath);
  }

  public async segment(input: DeepLabInput): Promise<RawSegmentationMap> {
    const model = await this.model;
    return tf.tidy(() => {
      const data = toInputTensor(input);
      return tf.squeeze(model.execute(data) as tf.Tensor);
    }) as RawSegmentationMap;
  }

  public async translate(
    rawSegmentationMap: RawSegmentationMap,
    canvas?: HTMLCanvasElement
  ): Promise<SegmentationData> {
    const [height, width] = rawSegmentationMap.shape;
    const colormap = getColormap(this.base);
    const translatedMapBuffer = tf.buffer([height, width, 3], 'int32');
    const mapData = (await rawSegmentationMap.array()) as number[][];
    const labels = new Set<Label>();
    for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
      for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
        const label: Label = mapData[columnIndex][rowIndex];
        labels.add(label);
        colormap[label].forEach((depth, channel) => {
          translatedMapBuffer.set(depth, columnIndex, rowIndex, channel);
        });
      }
    }

    const translatedMapTensor = translatedMapBuffer.toTensor() as tf.Tensor3D;

    const segmentationMap = await tf.browser.toPixels(
      translatedMapTensor,
      canvas
    );

    tf.dispose(translatedMapTensor);

    const labelNames = translateLabels(this.base);
    const legend: Legend = Array.from(labels).reduce(
      (accumulator, label) => ({
        ...accumulator,
        [labelNames[label]]: colormap[label],
      }),
      {}
    );

    return { legend, segmentationMap };
  }

  public async predict(
    input: DeepLabInput,
    canvas?: HTMLCanvasElement
  ): Promise<DeepLabOutput> {
    const rawSegmentationMap = await this.segment(input);

    const [height, width] = rawSegmentationMap.shape;
    const { legend, segmentationMap } = await this.translate(
      rawSegmentationMap,
      canvas
    );

    tf.dispose(rawSegmentationMap);

    return { legend, height, width, segmentationMap };
  }

  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    (await this.model).dispose();
  }
}
