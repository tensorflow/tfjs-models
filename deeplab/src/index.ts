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
import {DeepLabInput, DeepLabOutput, Label, Legend, SegmentationData, SemanticSegmentationBaseModel,} from './types';
import {getColormap, getLabels, toInputTensor} from './utils';

export {getLabels};
export async function load(
    base: SemanticSegmentationBaseModel = 'pascal', isQuantized = true) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js.` +
        ` If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if (['pascal', 'cityscapes', 'ade20k'].indexOf(base) === -1) {
    throw new Error(
        `SemanticSegmentation cannot be constructed ` +
        `with an invalid base model ${base}. ` +
        `Try one of 'pascal', 'cityscapes' and 'ade20k'.`);
  }

  const semanticSegmentation = new SemanticSegmentation(base);
  await semanticSegmentation.load();
  return semanticSegmentation;
}

export class SemanticSegmentation {
  private base: SemanticSegmentationBaseModel;
  private model: tf.GraphModel;
  private modelPath: string;
  constructor(base: SemanticSegmentationBaseModel, isQuantized = true) {
    this.base = base;

    this.modelPath = `${config['BASE_PATH']}/${
        isQuantized ? 'quantized/' : ''}${base}/model.json`;
  }

  public hasLoaded() {
    return !!this.model;
  }
  public async load() {
    this.model = await tf.loadGraphModel(this.modelPath);

    // Warmup the model.
    const result = await this.predict(tf.zeros([227, 227, 3])) as tf.Tensor;
    await result.data();
    result.dispose();
  }

  public async predict(input: DeepLabInput): Promise<tf.Tensor2D> {
    return tf.tidy(() => {
      const data = toInputTensor(input);
      return tf.squeeze(this.model.execute(data) as tf.Tensor);
    }) as tf.Tensor2D;
  }
  public async toSegmentationImage(
      rawSegmentationMap: tf.Tensor2D,
      canvas?: HTMLCanvasElement): Promise<SegmentationData> {
    const [height, width] = rawSegmentationMap.shape;
    const colormap = getColormap(this.base);
    const segmentationImageBuffer = tf.buffer([height, width, 3], 'int32');
    const mapData = (await rawSegmentationMap.array()) as number[][];
    const labels = new Set<Label>();
    for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
      for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
        const label: Label = mapData[columnIndex][rowIndex];
        labels.add(label);
        segmentationImageBuffer.set(
            colormap[label][0], columnIndex, rowIndex, 0);
        segmentationImageBuffer.set(
            colormap[label][1], columnIndex, rowIndex, 1);
        segmentationImageBuffer.set(
            colormap[label][2], columnIndex, rowIndex, 2);
      }
    }

    const segmentationImageTensor =
        segmentationImageBuffer.toTensor() as tf.Tensor3D;

    const segmentationMap =
        await tf.browser.toPixels(segmentationImageTensor, canvas);

    tf.dispose(segmentationImageTensor);

    const labelNames = getLabels(this.base);
    const legend: Legend = {};
    for (const label of Array.from(labels)) {
      legend[labelNames[label]] = colormap[label];
    }
    return {legend, segmentationMap};
  }

  public async segment(input: DeepLabInput, canvas?: HTMLCanvasElement):
      Promise<DeepLabOutput> {
    const rawSegmentationMap = await this.predict(input);

    const [height, width] = rawSegmentationMap.shape;
    const {legend, segmentationMap} =
        await this.toSegmentationImage(rawSegmentationMap, canvas);

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
