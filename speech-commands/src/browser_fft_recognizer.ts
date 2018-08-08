/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {loadMetadataJson} from './browser_fft_utils';
// tslint:disable-next-line:max-line-length
import {RecognizerCallback, RecognizerConfigParams, SpeechCommandRecognizer, SpeechCommandRecognizerResult, StreamingRecognitionConfig} from './types';

export class SpeechCommandBrowserFftRecognizer implements
    SpeechCommandRecognizer {
  // tslint:disable:max-line-length
  readonly DEFAULT_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-command-model-17w/model.json';
  readonly DEFAULT_METADATA_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-command-model-17w/metadata.json';
  // tslint:enable:max-line-length

  private readonly SAMPLE_RATE_HZ = 44100;
  private readonly FFT_SIZE = 1024;

  model: tf.Model;
  readonly parameters: RecognizerConfigParams;
  protected words: string[];
  private streaming: boolean;

  private nonBatchInputShape: [number, number, number];
  private elementsPerExample: number;

  constructor() {
    this.streaming = false;
    this.parameters = {
      sampleRateHz: this.SAMPLE_RATE_HZ,
      fftSize: this.FFT_SIZE,
      columnBufferLength: this.FFT_SIZE,
      columnHopLength: this.FFT_SIZE
    };
  }

  async startStreaming(
      callback: RecognizerCallback,
      config?: StreamingRecognitionConfig): Promise<void> {
    await this.ensureModelLoaded();

    // TODO(cais): Fill in.
    this.streaming = true;
  }

  async ensureModelLoaded() {
    if (this.model != null) {
      return;
    }

    const metadataJSON = await loadMetadataJson(this.DEFAULT_METADATA_JSON_URL);

    const model = await tf.loadModel(this.DEFAULT_MODEL_JSON_URL);
    // Check the validity of the model's input shape.
    if (model.inputs.length !== 1) {
      throw new Error(
          `Expected model to have 1 input, but got a model with ` +
          `${model.inputs.length} inputs`);
    }
    if (model.inputs[0].shape.length !== 4) {
      throw new Error(
          `Expected model to have an input shape of rank 4, ` +
          `but got an input shape of rank ${model.inputs[0].shape.length}`);
    }
    // Check the consistency between the word labels and the model's output
    // shape.
    const outputShape = model.outputShape as tf.Shape;
    if (outputShape.length !== 2) {
      throw new Error(
          `Expected loaded model to have an output shape of rank 2,` +
          `but received shape ${JSON.stringify(outputShape)}`);
    }
    if (outputShape[1] !== metadataJSON.words.length) {
      throw new Error(
          `Mismatch between the last dimension of model's output shape ` +
          `(${outputShape[1]}) and number of words ` +
          `(${metadataJSON.words.length}).`);
    }

    this.words = metadataJSON.words;
    this.model = model;
    this.nonBatchInputShape =
        this.model.inputs[0].shape.slice(1) as [number, number, number];
    this.elementsPerExample = 1;
    this.model.inputs[0].shape.slice(1).forEach(
        dimSize => this.elementsPerExample *= dimSize);

    const frameDurationMillis =
        this.parameters.columnBufferLength / this.parameters.sampleRateHz * 1e3;
    const numFrames = this.model.inputs[0].shape[1];
    this.parameters.spectrogramDurationMillis = numFrames * frameDurationMillis;
  }

  async stopStreaming(): Promise<void> {
    if (!this.streaming) {
      throw new Error(
          'This SpeechCommandBrowserFftRegonizer object is not ' +
          'currently streaming.');
    }
    // TODO(cais): Fill in.
  }

  isStreaming(): boolean {
    return this.streaming;
  }

  wordLabels(): string[] {
    return this.words;
  }

  params(): RecognizerConfigParams {
    return this.parameters;
  }

  async recognize(input: tf.Tensor|
                  Float32Array): Promise<SpeechCommandRecognizerResult> {
    await this.ensureModelLoaded();

    let numExamples: number;
    let inputTensor: tf.Tensor;
    let outTensor: tf.Tensor;
    if (input instanceof tf.Tensor) {
      // Check input shape.
      this.checkInputTensorShape(input);
      inputTensor = input;
      numExamples = input.shape[0];
    } else {
      input = input as Float32Array;
      if (input.length % this.elementsPerExample) {
        throw new Error(
            `The length of the input Float32Array ${input.length} ` +
            `is not divisible by the number of tensor elements per ` +
            `per example expected by the model ${this.elementsPerExample}.`);
      }

      numExamples = input.length / this.elementsPerExample;

      inputTensor = tf.tensor4d(input, [
        numExamples
      ].concat(this.nonBatchInputShape) as [number, number, number, number]);
    }

    outTensor = this.model.predict(inputTensor) as tf.Tensor;
    if (numExamples === 1) {
      return {scores: outTensor.dataSync() as Float32Array};
    } else {
      const unstacked = tf.unstack(outTensor);
      const scores = unstacked.map(item => item.dataSync() as Float32Array);
      return {scores};
    }
  }

  private checkInputTensorShape(input: tf.Tensor) {
    const expectedRank = this.model.inputs[0].shape.length;
    if (input.shape.length !== expectedRank) {
      throw new Error(
          `Expected input Tensor to have rank ${expectedRank}, ` +
          `but got rank ${input.shape.length} that differs `);
    }
    const nonBatchedShape = input.shape.slice(1);
    const expectedNonBatchShape = this.model.inputs[0].shape.slice(1);
    if (!tf.util.arraysEqual(nonBatchedShape, expectedNonBatchShape)) {
      throw new Error(
          `Expected input to have shape [null,${expectedNonBatchShape}], ` +
          `but got shape [null,${nonBatchedShape}]`);
    }
  }
}