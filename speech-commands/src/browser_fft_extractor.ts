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

/**
 * Audio FFT Feature Extractor based on Browser-Native FFT.
 */

import * as tf from '@tensorflow/tfjs';

// tslint:disable-next-line:max-line-length
import {getAudioContextConstructor, getAudioMediaStream, normalize} from './browser_fft_utils';
import {FeatureExtractor, RecognizerParams} from './types';

export type SpectrogramCallback = (x: tf.Tensor) => Promise<boolean>;

/**
 * Configurations for constructing BrowserFftFeatureExtractor.
 */
export interface BrowserFftFeatureExtractorConfig extends RecognizerParams {
  /**
   * Number of audio frames (i.e., frequency columns) per spectrogram.
   */
  numFramesPerSpectrogram: number;

  /**
   * Suppression period in milliseconds.
   *
   * How much time to rest (not call the spectrogramCallback) every time
   * a word with probability score above threshold is recognized.
   */
  suppressionTimeMillis: number;

  /**
   * A callback that is invoked every time a full spectrogram becomes
   * available.
   *
   * `x` is a single-example tf.Tensor instance that includes the batch
   * dimension.
   * The return value is assumed to be whether a flag for whether the
   * suppression period should initiate, e.g., when a word is recognized.
   */
  spectrogramCallback: SpectrogramCallback;

  /**
   * Truncate each spectrogram column at how many frequency points.
   *
   * If `null` or `undefined`, will do no truncation.
   */
  columnTruncateLength?: number;
}

/**
 * Audio feature extractor based on Browser-native FFT.
 *
 * Uses AudioContext and analyser node.
 */
export class BrowserFftFeatureExtractor implements FeatureExtractor {
  // Number of frames (i.e., columns) per spectrogram used for classification.
  readonly numFramesPerSpectrogram: number;

  // Audio sampling rate in Hz.
  readonly sampleRateHz: number;

  // The FFT length for each spectrogram column.
  readonly fftSize: number;

  // Truncation length for spectrogram columns.
  readonly columnTruncateLength: number;

  // Overlapping factor: the ratio between the temporal spacing between
  // consecutive spectrograms and the length of each individual spectrogram.
  readonly overlapFactor: number;

  protected readonly spectrogramCallback: SpectrogramCallback;

  private stream: MediaStream;
  // tslint:disable-next-line:no-any
  private audioContextConstructor: any;
  private audioContext: AudioContext;
  private analyser: AnalyserNode;

  private tracker: Tracker;

  private readonly ROTATING_BUFFER_SIZE_MULTIPLIER = 2;
  private freqData: Float32Array;
  private rotatingBufferNumFrames: number;
  private rotatingBuffer: Float32Array;

  private frameCount: number;
  // tslint:disable-next-line:no-any
  private frameIntervalTask: any;
  private frameDurationMillis: number;

  private suppressionTimeMillis: number;

  /**
   * Constructor of BrowserFftFeatureExtractor.
   *
   * @param config Required configuration object.
   */
  constructor(config: BrowserFftFeatureExtractorConfig) {
    if (config == null) {
      throw new Error(
          `Required configuration object is missing for ` +
          `BrowserFftFeatureExtractor constructor`);
    }

    if (config.spectrogramCallback == null) {
      throw new Error(`spectrogramCallback cannot be null or undefined`);
    }

    if (!(config.numFramesPerSpectrogram > 0)) {
      throw new Error(
          `Invalid value in numFramesPerSpectrogram: ` +
          `${config.numFramesPerSpectrogram}`);
    }

    if (config.suppressionTimeMillis < 0) {
      throw new Error(
          `Expected suppressionTimeMillis to be >= 0, ` +
          `but got ${config.suppressionTimeMillis}`);
    }
    this.suppressionTimeMillis = config.suppressionTimeMillis;

    this.spectrogramCallback = config.spectrogramCallback;
    this.numFramesPerSpectrogram = config.numFramesPerSpectrogram;
    this.sampleRateHz = config.sampleRateHz || 44100;
    this.fftSize = config.fftSize || 1024;
    this.frameDurationMillis = this.fftSize / this.sampleRateHz * 1e3;
    this.columnTruncateLength = config.columnTruncateLength || this.fftSize;
    const columnBufferLength = config.columnBufferLength || this.fftSize;
    const columnHopLength = config.columnHopLength || (this.fftSize / 2);
    this.overlapFactor = columnHopLength / columnBufferLength;

    if (!(this.overlapFactor > 0)) {
      throw new Error(
          `Invalid overlapFactor: ${this.overlapFactor}. ` +
          `Check your columnBufferLength and columnHopLength.`);
    }

    if (this.columnTruncateLength > this.fftSize) {
      throw new Error(
          `columnTruncateLength ${this.columnTruncateLength} exceeds ` +
          `fftSize (${this.fftSize}).`);
    }

    this.audioContextConstructor = getAudioContextConstructor();
  }

  async start(samples?: Float32Array): Promise<Float32Array[]|void> {
    if (this.frameIntervalTask != null) {
      throw new Error(
          'Cannot start already-started BrowserFftFeatureExtractor');
    }

    this.stream = await getAudioMediaStream();
    this.audioContext = new this.audioContextConstructor() as AudioContext;
    if (this.audioContext.sampleRate !== this.sampleRateHz) {
      console.warn(
          `Mismatch in sampling rate: ` +
          `Expected: ${this.sampleRateHz}; ` +
          `Actual: ${this.audioContext.sampleRate}`);
    }
    const streamSource = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize * 2;
    this.analyser.smoothingTimeConstant = 0.0;
    streamSource.connect(this.analyser);

    this.freqData = new Float32Array(this.fftSize);
    this.rotatingBufferNumFrames =
        this.numFramesPerSpectrogram * this.ROTATING_BUFFER_SIZE_MULTIPLIER;
    const rotatingBufferSize =
        this.columnTruncateLength * this.rotatingBufferNumFrames;
    this.rotatingBuffer = new Float32Array(rotatingBufferSize);

    this.frameCount = 0;

    this.tracker = new Tracker(
        Math.round(this.numFramesPerSpectrogram * this.overlapFactor),
        Math.round(this.suppressionTimeMillis / this.frameDurationMillis));
    this.frameIntervalTask = setInterval(
        this.onAudioFrame.bind(this), this.fftSize / this.sampleRateHz * 1e3);
  }

  private async onAudioFrame() {
    this.analyser.getFloatFrequencyData(this.freqData);
    if (this.freqData[0] === -Infinity) {
      console.warn(`No signal (frame #${this.frameCount})`);
      return;
    }

    const freqDataSlice = this.freqData.slice(0, this.columnTruncateLength);
    const bufferPos = this.frameCount % this.rotatingBufferNumFrames;
    this.rotatingBuffer.set(
        freqDataSlice, bufferPos * this.columnTruncateLength);
    this.frameCount++;

    const shouldFire = this.tracker.tick();
    if (shouldFire) {
      const freqData = getFrequencyDataFromRotatingBuffer(
          this.rotatingBuffer, this.numFramesPerSpectrogram,
          this.columnTruncateLength,
          this.frameCount - this.numFramesPerSpectrogram);
      const inputTensor = getInputTensorFromFrequencyData(
          freqData, this.numFramesPerSpectrogram, this.columnTruncateLength);
      const shouldRest = await this.spectrogramCallback(inputTensor);
      if (shouldRest) {
        this.tracker.suppress();
      }
      inputTensor.dispose();
    }
  }

  async stop(): Promise<void> {
    if (this.frameIntervalTask == null) {
      throw new Error(
          'Cannot stop because there is no ongoing streaming activity.');
    }
    clearInterval(this.frameIntervalTask);
    this.frameIntervalTask = null;
    this.analyser.disconnect();
    this.audioContext.close();
  }

  setConfig(params: RecognizerParams) {
    throw new Error(
        'setConfig() is not implemented for BrowserFftFeatureExtractor.');
  }

  getFeatures(): Float32Array[] {
    throw new Error(
        'getFeatures() is not implemented for ' +
        'BrowserFftFeatureExtractor. Use the spectrogramCallback ' +
        'field of the constructor config instead.');
  }
}

export function getFrequencyDataFromRotatingBuffer(
    rotatingBuffer: Float32Array, numFrames: number, fftLength: number,
    frameCount: number): Float32Array {
  const size = numFrames * fftLength;
  const freqData = new Float32Array(size);

  const rotatingBufferSize = rotatingBuffer.length;
  const rotatingBufferNumFrames = rotatingBufferSize / fftLength;
  while (frameCount < 0) {
    frameCount += rotatingBufferNumFrames;
  }
  const indexBegin = (frameCount % rotatingBufferNumFrames) * fftLength;
  const indexEnd = indexBegin + size;

  for (let i = indexBegin; i < indexEnd; ++i) {
    freqData[i - indexBegin] = rotatingBuffer[i % rotatingBufferSize];
  }
  return freqData;
}

export function getInputTensorFromFrequencyData(
    freqData: Float32Array, numFrames: number, fftLength: number,
    toNormalize = true): tf.Tensor {
  return tf.tidy(() => {
    const size = freqData.length;
    const tensorBuffer = tf.buffer([size]);
    for (let i = 0; i < freqData.length; ++i) {
      tensorBuffer.set(freqData[i], i);
    }
    const output =
        tensorBuffer.toTensor().reshape([1, numFrames, fftLength, 1]);
    return toNormalize ? normalize(output) : output;
  });
}

/**
 * A class that manages the firing of events based on periods
 * and suppression time.
 */
export class Tracker {
  readonly period: number;
  readonly suppressionTime: number;

  private counter: number;
  private suppressionOnset: number;

  /**
   * Constructor of Tracker.
   *
   * @param period The event-firing period, in number of frames.
   * @param suppressionPeriod The suppression period, in number of frames.
   */
  constructor(period: number, suppressionPeriod: number) {
    this.period = period;
    this.suppressionTime = suppressionPeriod == null ? 0 : suppressionPeriod;
    this.counter = 0;

    tf.util.assert(
        this.period > 0,
        `Expected period to be positive, but got ${this.period}`);
  }

  /**
   * Mark a frame.
   *
   * @returns Whether the event should be fired at the current frame.
   */
  tick(): boolean {
    this.counter++;
    const shouldFire = (this.counter % this.period === 0) &&
        (this.suppressionOnset == null ||
         this.counter - this.suppressionOnset > this.suppressionTime);
    return shouldFire;
  }

  /**
   * Order the beginning of a supression period.
   */
  suppress() {
    this.suppressionOnset = this.counter;
  }
}
