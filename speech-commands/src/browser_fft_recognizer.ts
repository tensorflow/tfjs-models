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

// tslint:disable:max-line-length
import {BrowserFftFeatureExtractor, SpectrogramCallback} from './browser_fft_extractor';
import {loadMetadataJson} from './browser_fft_utils';
import {RecognizerCallback, RecognizerConfigParams, SpectrogramData, SpeechCommandRecognizer, SpeechCommandRecognizerResult, StreamingRecognitionConfig} from './types';
// tslint:enable:max-line-length

/**
 * Speech-Command Recognizer using browser-native (WebAudio) spectral featutres.
 */
export class BrowserFftSpeechCommandRecognizer implements
    SpeechCommandRecognizer {
  // tslint:disable:max-line-length
  readonly DEFAULT_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-commands-models/19w/model.json';
  readonly DEFAULT_METADATA_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-commands-models/19w/metadata.json';
  // tslint:enable:max-line-length

  private readonly SAMPLE_RATE_HZ = 44100;
  private readonly FFT_SIZE = 1024;

  model: tf.Model;
  readonly parameters: RecognizerConfigParams;
  protected words: string[];
  private streaming: boolean;

  private nonBatchInputShape: [number, number, number];
  private elementsPerExample: number;
  private audioDataExtractor: BrowserFftFeatureExtractor;

  /**
   * Constructor of BrowserFftSpeechCommandRecognizer.
   */
  constructor() {
    // TODO(cais): Call this constructor in a factory function.
    this.streaming = false;
    this.parameters = {
      sampleRateHz: this.SAMPLE_RATE_HZ,
      fftSize: this.FFT_SIZE,
      columnBufferLength: this.FFT_SIZE,
    };
  }

  /**
   * Start streaming recognition.
   *
   * To stop the recognition, use `stopStreaming()`.
   *
   * Example: TODO(cais): Add exapmle code snippet.
   *
   * @param callback The callback invoked whenever a word is recognized
   *   with a probability score greater than `config.probabilityThreshold`.
   *   It has the signature:
   *     (result: SpeechCommandRecognizerResult) => Promise<void>
   *   wherein result has the two fields:
   *   - scores: A Float32Array that contains the probability scores for all
   *     the words.
   *   - spectrogram: The spectrogram data, provided only if
   *     `config.includeSpectrogram` is `true`.
   * @param config The configurations for the streaming recognition to
   *   be started.
   * @throws Error, if streaming recognition is already started or
   *   if `config` contains invalid values.
   */
  async startStreaming(
      callback: RecognizerCallback,
      config?: StreamingRecognitionConfig): Promise<void> {
    if (this.streaming) {
      throw new Error(
          'Cannot start streaming again when streaming is ongoing.');
    }

    await this.ensureModelLoaded();

    if (config == null) {
      config = {};
    }
    const probabilityThreshold =
        config.probabilityThreshold == null ? 0 : config.probabilityThreshold;
    tf.util.assert(
        probabilityThreshold >= 0 && probabilityThreshold <= 1,
        `Invalid probabilityThreshold value: ${probabilityThreshold}`);

    const overlapFactor =
        config.overlapFactor == null ? 0.5 : config.overlapFactor;
    tf.util.assert(
        overlapFactor >= 0 && overlapFactor < 1,
        `Expected overlapFactor to be >= 0 and < 1, but got ${overlapFactor}`);
    this.parameters.columnHopLength =
        Math.round(this.FFT_SIZE * (1 - overlapFactor));

    const spectrogramCallback: SpectrogramCallback = (x: tf.Tensor) => {
      return tf.tidy(() => {
        const y = this.model.predict(x) as tf.Tensor;
        const scores = y.dataSync() as Float32Array;
        const maxScore = Math.max(...scores);
        if (maxScore < probabilityThreshold) {
          return false;
        } else {
          let spectrogram: SpectrogramData = undefined;
          if (config.includeSpectrogram) {
            spectrogram = {
              data: x.dataSync() as Float32Array,
              frameSize: this.nonBatchInputShape[1],
            };
          }
          callback({scores, spectrogram});
          return true;
        }
      });
    };

    this.audioDataExtractor = new BrowserFftFeatureExtractor({
      sampleRateHz: this.parameters.sampleRateHz,
      columnBufferLength: this.parameters.columnBufferLength,
      columnHopLength: this.parameters.columnHopLength,
      numFramesPerSpectrogram: this.nonBatchInputShape[0],
      columnTruncateLength: this.nonBatchInputShape[1],
      spectrogramCallback
    });

    await this.audioDataExtractor.start();

    this.streaming = true;
  }

  /**
   * Load the underlying tf.Model instance and associated metadata.
   *
   * If the model and the metadata are already loaded, do nothing.
   */
  async ensureModelLoaded() {
    if (this.model != null) {
      return;
    }

    await this.ensureMetadataLoaded();

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
    if (model.inputs[0].shape[3] !== 1) {
      throw new Error(
          `Expected model to have an input shape with 1 as the last ` +
          `dimension, but got input shape` +
          `${JSON.stringify(model.inputs[0].shape[3])}}`);
    }
    // Check the consistency between the word labels and the model's output
    // shape.
    const outputShape = model.outputShape as tf.Shape;
    if (outputShape.length !== 2) {
      throw new Error(
          `Expected loaded model to have an output shape of rank 2,` +
          `but received shape ${JSON.stringify(outputShape)}`);
    }
    if (outputShape[1] !== this.words.length) {
      throw new Error(
          `Mismatch between the last dimension of model's output shape ` +
          `(${outputShape[1]}) and number of words ` +
          `(${this.words.length}).`);
    }

    this.model = model;
    this.nonBatchInputShape =
        this.model.inputs[0].shape.slice(1) as [number, number, number];
    this.elementsPerExample = 1;
    this.model.inputs[0].shape.slice(1).forEach(
        dimSize => this.elementsPerExample *= dimSize);

    this.warmUpModel();

    const frameDurationMillis =
        this.parameters.columnBufferLength / this.parameters.sampleRateHz * 1e3;
    const numFrames = this.model.inputs[0].shape[1];
    this.parameters.spectrogramDurationMillis = numFrames * frameDurationMillis;
  }

  private warmUpModel() {
    tf.tidy(() => {
      const x = tf.zeros([1].concat(this.nonBatchInputShape));
      for (let i = 0; i < 3; ++i) {
        this.model.predict(x);
      }
    });
  }

  private async ensureMetadataLoaded() {
    if (this.words != null) {
      return;
    }
    const metadataJSON = await loadMetadataJson(this.DEFAULT_METADATA_JSON_URL);
    this.words = metadataJSON.words;
  }

  /**
   * Stop streaming recognition.
   *
   * @throws Error if there is not ongoing streaming recognition.
   */
  async stopStreaming(): Promise<void> {
    if (!this.streaming) {
      throw new Error('Cannot stop streaming when streaming is not ongoing.');
    }
    await this.audioDataExtractor.stop();
    this.streaming = false;
  }

  /**
   * Check if streaming recognition is ongoing.
   */
  isStreaming(): boolean {
    return this.streaming;
  }

  /**
   * Get the array of word labels.
   *
   * @throws Error If this model is called before the model is loaded.
   */
  wordLabels(): string[] {
    if (this.model == null) {
      throw new Error(
          'Model is not loaded yet. Call ensureModelLoaded() or ' +
          'use the model for recognition with startStreaming() or ' +
          'recognize() first.');
    }
    return this.words;
  }

  /**
   * Get the parameters of this instance of BrowserFftSpeechCommandRecognizer.
   *
   * @returns Parameters of this instance.
   */
  params(): RecognizerConfigParams {
    return this.parameters;
  }

  /**
   * Get the input shape of the underlying tf.Model.
   *
   * @returns The input shape.
   */
  modelInputShape(): tf.Shape {
    if (this.model == null) {
      throw new Error(
          'Model has not be loaded yet. Load model by calling ' +
          'ensureModelLoaded(), recognizer(), or startStreaming().');
    }
    return this.model.inputs[0].shape;
  }

  /**
   * Run offline (non-streaming) recognition on a spectrogram.
   *
   * @param input Spectrogram. Either a `tf.Tensor` of a `Float32Array`.
   *   - If a `tf.Tensor`, must be rank-4 and match the model's expected
   *     input shape in 2nd dimension (# of spectrogram columns), the 3rd
   *     dimension (# of frequency-domain points per column), and the 4th
   *     dimension (always 1). The 1st dimension can be 1, for single-example
   *     recogntion, or any value >1, for batched recognition.
   *   - If a `Float32Array`, must have a length divisible by the number
   *     of elements per spectrogram, i.e.,
   *     (# of spectrogram columns) * (# of frequency-domain points per column).
   * @returns Result of the recognition, with the following field:
   *   scores:
   *   - A `Float32Array` if there is only one input exapmle.
   *   - An `Array` of `Float32Array`, if there are multiple input examples.
   */
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
      // `input` is a `Float32Array`.
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
      const unstacked = tf.unstack(outTensor) as tf.Tensor[];
      const scores = unstacked.map(item => item.dataSync() as Float32Array);
      tf.dispose(unstacked);
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
