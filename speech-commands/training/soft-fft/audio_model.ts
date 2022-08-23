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

import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import * as fs from 'fs';
/// <reference path="./types/node-wav.d.ts" />
import * as wav from 'node-wav';
import * as path from 'path';

import {Dataset} from './dataset';

import {WavFileFeatureExtractor} from './wav_file_feature_extractor';

/**
 * Audio Model that creates tf.Model for a fix amount of labels. It requires a
 * feature extractor to convert the audio stream into input tensors for the
 * internal tf.Model.
 * It provide datasets loading, training, and model saving functions.
 */
export class AudioModel {
  private model: tfl.LayersModel;

  /**
   *
   * @param inputShape Input tensor shape.
   * @param labels Audio command label list
   * @param dataset Dataset class to store the loaded data.
   * @param featureExtractor converter to extractor features from audio stream
   * as input tensors
   */
  constructor(
      inputShape: number[], private labels: string[], private dataset: Dataset,
      private featureExtractor: WavFileFeatureExtractor) {
    this.featureExtractor.config({
      melCount: 40,
      bufferLength: 480,
      hopLength: 160,
      targetSr: 16000,
      isMfccEnabled: true,
      duration: 1.0
    });

    this.model = this.createModel(inputShape);
  }

  private createModel(inputShape: number[]): tfl.LayersModel {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2d(
        {filters: 8, kernelSize: [4, 2], activation: 'relu', inputShape}));
    model.add(tfl.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tfl.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tfl.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tfl.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tfl.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tfl.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tfl.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 2]}));
    model.add(tfl.layers.flatten({}));
    model.add(tfl.layers.dropout({rate: 0.25}));
    model.add(tfl.layers.dense({units: 2000, activation: 'relu'}));
    model.add(tfl.layers.dropout({rate: 0.5}));
    model.add(
        tfl.layers.dense({units: this.labels.length, activation: 'softmax'}));

    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.sgd(0.01),
      metrics: ['accuracy']
    });
    model.summary();
    return model;
  }

  /**
   * Load all dataset for the root directory, all the subdirectories that have
   * matching name to the entries in model label list, contained audio files
   * will be converted to input tensors and stored in the dataset for training.
   * @param dir The root directory of the audio dataset
   * @param callback Callback function for display training logs
   */
  async loadAll(dir: string, callback: Function) {
    const promises = [];
    this.labels.forEach(async (label, index) => {
      callback(`loading label: ${label} (${index})`);
      promises.push(
          this.loadDataArray(path.resolve(dir, label), callback).then(v => {
            callback(`finished loading label: ${label} (${index})`, true);
            return [v, index];
          }));
    });

    let allSpecs = await Promise.all(promises);
    allSpecs = allSpecs
                   .map((specs, i) => {
                     const index = specs[1];
                     return specs[0].map(spec => [spec, index]);
                   })
                   .reduce((acc, currentValue) => acc.concat(currentValue), []);

    tf.util.shuffle(allSpecs);
    const specs = allSpecs.map(spec => spec[0]);
    const labels = allSpecs.map(spec => spec[1]);
    this.dataset.addExamples(
        this.melSpectrogramToInput(specs),
        tf.oneHot(labels, this.labels.length));
  }

  /**
   * Load one dataset from directory, all contained audio files
   * will be converted to input tensors and stored in the dataset for training.
   * @param dir The directory of the audio dataset
   * @param label The label for the audio dataset
   * @param callback Callback function for display training logs
   */
  async loadData(dir: string, label: string, callback: Function) {
    const index = this.labels.indexOf(label);
    const specs = await this.loadDataArray(dir, callback);
    this.dataset.addExamples(
        this.melSpectrogramToInput(specs),
        tf.oneHot(tf.fill([specs.length], index, 'int32'), this.labels.length));
  }

  private loadDataArray(dir: string, callback: Function) {
    return new Promise<Float32Array[][]>((resolve, reject) => {
      fs.readdir(dir, (err, filenames) => {
        if (err) {
          reject(err);
        }
        let specs: Float32Array[][] = [];
        filenames.forEach((filename) => {
          callback('decoding ' + dir + '/' + filename + '...');
          const spec = this.splitSpecs(this.decode(dir + '/' + filename));
          if (!!spec) {
            specs = specs.concat(spec);
          }
          callback('decoding ' + dir + '/' + filename + '...done');
        });
        resolve(specs);
      });
    });
  }

  private decode(filename: string) {
    const result = wav.decode(fs.readFileSync(filename));
    return this.featureExtractor.start(result.channelData[0]);
  }

  /**
   * Train the model for stored dataset. The method call be called multiple
   * times.
   * @param epochs iteration of the training
   * @param trainCallback
   */
  async train(epochs?: number, trainCallback?: tfl.CustomCallbackArgs) {
    return this.model.fit(this.dataset.xs, this.dataset.ys, {
      batchSize: 64,
      epochs: epochs || 100,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: trainCallback
    });
  }

  /**
   * Save the model to the specified directory.
   * @param dir Directory to store the model.
   */
  save(dir: string): Promise<tf.io.SaveResult> {
    return this.model.save('file://' + dir);
  }

  /**
   * Return the size of the dataset in string.
   */
  size(): string {
    return this.dataset.xs ?
        `xs: ${this.dataset.xs.shape} ys: ${this.dataset.ys.shape}` :
        '0';
  }

  private splitSpecs(spec: Float32Array[]) {
    if (spec.length >= 98) {
      const output = [];
      for (let i = 0; i <= (spec.length - 98); i += 32) {
        output.push(spec.slice(i, i + 98));
      }
      return output;
    }
    return undefined;
  }

  private melSpectrogramToInput(specs: Float32Array[][]): tf.Tensor {
    // Flatten this spectrogram into a 2D array.
    const batch = specs.length;
    const times = specs[0].length;
    const freqs = specs[0][0].length;
    const data = new Float32Array(batch * times * freqs);
    console.log(data.length);
    for (let j = 0; j < batch; j++) {
      const spec = specs[j];
      for (let i = 0; i < times; i++) {
        const mel = spec[i];
        const offset = j * freqs * times + i * freqs;
        data.set(mel, offset);
      }
    }
    // Normalize the whole input to be in [0, 1].
    const shape: [number, number, number, number] = [batch, times, freqs, 1];
    // this.normalizeInPlace(data, 0, 1);
    return tf.tensor4d(data, shape);
  }
}
