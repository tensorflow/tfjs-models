import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
/// <reference path="./types/node-wav.d.ts" />
import * as wav from 'node-wav';
import * as path from 'path';

import {Dataset} from './dataset';

import {WavFileFeatureExtractor} from './wav_file_feature_extractor';

export class AudioModel {
  private model: tf.Model;
  private dataset: Dataset;
  private featureExtractor: WavFileFeatureExtractor;

  constructor(inputShape: number[], private labels: string[]) {
    console.log(tf.getBackend());
    console.log(inputShape);
    console.log(labels.length);
    this.dataset = new Dataset(labels.length);
    this.featureExtractor = new WavFileFeatureExtractor();
    this.featureExtractor.config({
      melCount: 40,
      bufferLength: 480,
      hopLength: 160,
      targetSr: 16000,
      isMfccEnabled: true,
      duration: 1.0
    });
    const model = tf.sequential();
    model.add(tf.layers.conv2d(
        {filters: 8, kernelSize: [4, 2], activation: 'relu', inputShape}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d(
        {filters: 32, kernelSize: [4, 2], activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 2]}));
    model.add(tf.layers.flatten({}));
    model.add(tf.layers.dense({units: 2000, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.5}));
    model.add(tf.layers.dense({units: labels.length, activation: 'softmax'}));

    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.sgd(0.01),
      metrics: ['accuracy']
    });
    model.summary();
    this.model = model;
  }

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
  decode(filename: string) {
    const result = wav.decode(fs.readFileSync(filename));
    return this.featureExtractor.start(result.channelData[0]);
  }

  train(epochs?: number, trainCallback?: tf.CustomCallbackConfig) {
    return this.model.fit(this.dataset.xs, this.dataset.ys, {
      batchSize: 64,
      epochs: epochs || 100,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: trainCallback
    });
  }

  save(filename: string) {
    return this.model.save('file://' + filename);
  }

  size() {
    return this.dataset.xs ?
        `xs: ${this.dataset.xs.shape} ys: ${this.dataset.ys.shape}` :
        0;
  }

  splitSpecs(spec: Float32Array[]) {
    if (spec.length >= 98) {
      const output = [];
      for (let i = 0; i <= (spec.length - 98); i += 32) {
        output.push(spec.slice(i, i + 98));
      }
      return output;
    }
    return undefined;
  }

  melSpectrogramToInput(specs: Float32Array[][]): tf.Tensor {
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
