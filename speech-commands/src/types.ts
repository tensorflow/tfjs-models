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

/**
 * This file defines the interfaces related to SpeechCommandRecognizer.
 */

export type FFT_TYPE = 'BROWSER_FFT'|'SOFT_FFT';

export type RecognizerCallback = (result: SpeechCommandRecognizerResult) =>
    Promise<void>;

export interface SpeechCommandRecognizer<FFT_TYPE> {
  // Start recognition in a streaming fashion.
  //
  // Args:
  //   callback: the callback that will be invoked every time
  //     a recognition result is available.
  //   options: optional configuration.
  // Throws:
  //   Error if there is already ongoing streaming recognition.
  startStreaming(
      callback: RecognizerCallback,
      config?: StreamingRecognitionConfig): Promise<void>;

  // Stop the ongoing streaming recognition (if any).
  //
  // Throws:
  //   Error if no streaming recognition is ongoing.
  stopStreaming(): Promise<void>;

  // Check if this instance is currently performing
  // streaming recognition.
  isStreaming(): boolean;

  // Recognize a single example of audio.
  //
  // Args:
  //   input: tf.Tensor of Float32Array. If a tf.Tensor,
  //     must match the input shape of the underlying
  //     tf.Model. If a Float32Array, the length must be
  //     equal to (the model’s required FFT length) *
  //     (the model’s required frame count).
  // Returns: recognition result: the probability scores.
  // Throws: Error on incorrect shape or length.
  recognize(input: tf.Tensor|Float32Array): SpeechCommandRecognizerResult;

  // Getter word labels.
  readonly wordLabels: string;

  // Get the required number of frames.
  readonly params: RecognizerConfigParams;
}

export interface SpectrogramData {
  // The float32 data for the spectrogram.
  data: Float32Array;

  // Number of points per frame, i.e., FFT length per frame.
  frameSize: number;
}

export interface SpeechCommandRecognizerResult {
  // Probability scores for the words.
  scores: Float32Array;

  // Optional spectrogram data.
  spectrogram?: SpectrogramData;
}

export interface StreamingRecognitionConfig {
  // Overlap factor. Must be a number between >=0 and <1.
  // Defaults to 0.5.
  // For example, if the model takes a frame length of 1000 ms,
  // and if overlap factor is 0.4, there will be a 400-ms
  // overlap between two successive frames, i.e., frames
  // will be taken every 600 ms.
  overlapFactor?: number;

  // minimum samples of the same label for reliable prediction.
  minSamples?: number;

  // amount to time in ms to suppress recognizer after a word is recognized.
  suppressionTimeInMs?: number;
}

export interface RecognizerConfigParams {
  // audio sample window size per spectrogram column.
  columnBufferLength?: number;

  // audio sample window hopping size between two consecutive spectrogram
  // columns.
  columnHopLength?: number;

  // total duration per spectragram.
  SpectrogramDurationMillis?: number;

  // FFT encoding size per spectrogram column.
  fftSize?: number;

  // post FFT filter size for spectorgram column.
  filterSize?: number;

  // sampling rate in Hz.
  sampleRateHz?: number;
}

export interface FeatureExtractor {
  // config the feature extractor.
  setConfig(params: RecognizerConfigParams): void;

  // start the feature extraction from the audio samples.
  start(samples?: Float32Array): Promise<Float32Array[]>|void;

  // stop the feature extraction.
  stop(): void;

  // return the extractor features collected since last call.
  getFeatures(): Float32Array[];
}
