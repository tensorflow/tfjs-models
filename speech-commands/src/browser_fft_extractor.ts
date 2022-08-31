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

/**
 * Audio FFT Feature Extractor based on Browser-Native FFT.
 */

import * as tf from '@tensorflow/tfjs-core';

import {getAudioContextConstructor, getAudioMediaStream} from './browser_fft_utils';
import {FeatureExtractor, RecognizerParams} from './types';

export type SpectrogramCallback = (freqData: tf.Tensor, timeData?: tf.Tensor) =>
    Promise<boolean>;

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

  /**
   * Overlap factor. Must be >=0 and <1.
   * For example, if the model takes a frame length of 1000 ms,
   * and if overlap factor is 0.4, there will be a 400ms
   * overlap between two successive frames, i.e., frames
   * will be taken every 600 ms.
   */
  overlapFactor: number;

  /**
   * Whether to collect the raw time-domain audio waveform in addition to the
   * spectrogram.
   *
   * Default: `false`.
   */
  includeRawAudio?: boolean;
}

/**
 * Audio feature extractor based on Browser-native FFT.
 *
 * Uses AudioContext and analyser node.
 */
export class BrowserFftFeatureExtractor implements FeatureExtractor {
  // Number of frames (i.e., columns) per spectrogram used for classification.
  readonly numFrames: number;

  // Audio sampling rate in Hz.
  readonly sampleRateHz: number;

  // The FFT length for each spectrogram column.
  readonly fftSize: number;

  // Truncation length for spectrogram columns.
  readonly columnTruncateLength: number;

  // Overlapping factor: the ratio between the temporal spacing between
  // consecutive spectrograms and the length of each individual spectrogram.
  readonly overlapFactor: number;
  readonly includeRawAudio: boolean;

  private readonly spectrogramCallback: SpectrogramCallback;

  private stream: MediaStream;
  // tslint:disable-next-line:no-any
  private audioContextConstructor: any;
  private audioContext: AudioContext;
  private analyser: AnalyserNode;
  private tracker: Tracker;
  private freqData: Float32Array;
  private timeData: Float32Array;
  private freqDataQueue: Float32Array[];
  private timeDataQueue: Float32Array[];
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
    this.numFrames = config.numFramesPerSpectrogram;
    this.sampleRateHz = config.sampleRateHz || 44100;
    this.fftSize = config.fftSize || 1024;
    this.frameDurationMillis = this.fftSize / this.sampleRateHz * 1e3;
    this.columnTruncateLength = config.columnTruncateLength || this.fftSize;
    this.overlapFactor = config.overlapFactor;
    this.includeRawAudio = config.includeRawAudio;

    tf.util.assert(
        this.overlapFactor >= 0 && this.overlapFactor < 1,
        () => `Expected overlapFactor to be >= 0 and < 1, ` +
            `but got ${this.overlapFactor}`);

    if (this.columnTruncateLength > this.fftSize) {
      throw new Error(
          `columnTruncateLength ${this.columnTruncateLength} exceeds ` +
          `fftSize (${this.fftSize}).`);
    }

    this.audioContextConstructor = getAudioContextConstructor();
  }

  async start(audioTrackConstraints?: MediaTrackConstraints):
      Promise<Float32Array[]|void> {
    if (this.frameIntervalTask != null) {
      throw new Error(
          'Cannot start already-started BrowserFftFeatureExtractor');
    }

    this.stream = await getAudioMediaStream(audioTrackConstraints);
    this.audioContext = new this.audioContextConstructor(
                            {sampleRate: this.sampleRateHz}) as AudioContext;

    const streamSource = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize * 2;
    this.analyser.smoothingTimeConstant = 0.0;
    streamSource.connect(this.analyser);
    // Reset the queue.
    this.freqDataQueue = [];
    this.freqData = new Float32Array(this.fftSize);
    if (this.includeRawAudio) {
      this.timeDataQueue = [];
      this.timeData = new Float32Array(this.fftSize);
    }
    const period =
        Math.max(1, Math.round(this.numFrames * (1 - this.overlapFactor)));
    this.tracker = new Tracker(
        period,
        Math.round(this.suppressionTimeMillis / this.frameDurationMillis));
    this.frameIntervalTask = setInterval(
        this.onAudioFrame.bind(this), this.fftSize / this.sampleRateHz * 1e3);
  }

  private async onAudioFrame() {
    this.analyser.getFloatFrequencyData(this.freqData);
    if (this.freqData[0] === -Infinity) {
      return;
    }

    this.freqDataQueue.push(this.freqData.slice(0, this.columnTruncateLength));
    if (this.includeRawAudio) {
      this.analyser.getFloatTimeDomainData(this.timeData);
      this.timeDataQueue.push(this.timeData.slice());
    }
    if (this.freqDataQueue.length > this.numFrames) {
      // Drop the oldest frame (least recent).
      this.freqDataQueue.shift();
    }
    const shouldFire = this.tracker.tick();
    if (shouldFire) {
      const freqData = flattenQueue(this.freqDataQueue);
      const freqDataTensor = getInputTensorFromFrequencyData(
          freqData, [1, this.numFrames, this.columnTruncateLength, 1]);
      let timeDataTensor: tf.Tensor;
      if (this.includeRawAudio) {
        const timeData = flattenQueue(this.timeDataQueue);
        timeDataTensor = getInputTensorFromFrequencyData(
            timeData, [1, this.numFrames * this.fftSize]);
      }
      const shouldRest =
          await this.spectrogramCallback(freqDataTensor, timeDataTensor);
      if (shouldRest) {
        this.tracker.suppress();
      }
      tf.dispose([freqDataTensor, timeDataTensor]);
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
    if (this.stream != null && this.stream.getTracks().length > 0) {
      this.stream.getTracks()[0].stop();
    }
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

export function flattenQueue(queue: Float32Array[]): Float32Array {
  const frameSize = queue[0].length;
  const freqData = new Float32Array(queue.length * frameSize);
  queue.forEach((data, i) => freqData.set(data, i * frameSize));
  return freqData;
}

export function getInputTensorFromFrequencyData(
    freqData: Float32Array, shape: number[]): tf.Tensor {
  const vals = new Float32Array(tf.util.sizeFromShape(shape));
  // If the data is less than the output shape, the rest is padded with zeros.
  vals.set(freqData, vals.length - freqData.length);
  return tf.tensor(vals, shape);
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
        () => `Expected period to be positive, but got ${this.period}`);
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
