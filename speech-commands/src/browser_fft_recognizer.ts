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

import {RecognizerCallback, RecognizerConfigParams, SpeechCommandRecognizer, SpeechCommandRecognizerResult, StreamingRecognitionConfig} from './types';



export class SpeechCommandBrowserFftRecognizer implements
    SpeechCommandRecognizer {
  readonly DEFAULT_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-command-model-17w/model.json';
  readonly DEFAULT_METADATA_JSON_URL =
      'https://storage.googleapis.com/tfjs-speech-command-model-17w/metadata.json';

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
    this.model = await tf.loadModel(this.DEFAULT_MODEL_JSON_URL);
    const frameDurationMillis =
        this.params.columnBufferLength / this.params.sampleRateHz * 1e3;
    const numFrames = this.model.inputs[0].shape[1];
    this.params.spectrogramDurationMillis = numFrames * frameDurationMillis;

    const metadataJSON =
        await (await fetch(this.DEFAULT_METADATA_JSON_URL)).json();
    this.words = metadataJSON.words;
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

  recognize(input: tf.Tensor|Float32Array): SpeechCommandRecognizerResult {
    return null;  // TODO(cais).
  }

  get wordLabels(): string[] {
    return this.words;
  }
}