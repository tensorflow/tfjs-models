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

  readonly params: RecognizerConfigParams;
  protected words: string[];
  protected model: tf.Model;
  private streaming: boolean;

  constructor() {
    this.streaming = false;
    this.params = {
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

    const frameDurationMillis =
        this.params.columnBufferLength / this.params.sampleRateHz * 1e3;
    const numFrames = this.model.inputs[0].shape[1];
    this.params.spectrogramDurationMillis = numFrames * frameDurationMillis;
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

  recognize(input: tf.Tensor|Float32Array): SpeechCommandRecognizerResult {
    return null;  // TODO(cais).
  }
}