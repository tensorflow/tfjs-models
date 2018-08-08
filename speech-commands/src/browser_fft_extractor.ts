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

import {normalize} from './browser_fft_utils';
import {FeatureExtractor, RecognizerConfigParams} from './types';

export type SpectrogramCallback = (x: tf.Tensor) => boolean;

export interface BrowserFftFeatureExtractorConfig extends
    RecognizerConfigParams {
  /**
   * Number of audio frames (i.e., frequency columns) per spectrogram.
   */
  numFramesPerSpectrogram: number;

  /**
   * A callback that is invoked every time a full spectrogram becomes
   * available.
   *
   * `x` is a single-example tf.Tensor instance that includes the batch
   * dimension.
   * The return value is assumed to be whether a flag for whether the
   * refractory period should initiate, e.g., when a word is recognized.
   */
  spectrogramCallback: SpectrogramCallback;

  /**
   * Truncate each spectrogram column at how many frequency points.
   *
   * If `null` or `undefined`, will do no truncation.
   */
  columnTruncateLength?: number;
}

export class BrowserFftFeatureExtractor implements FeatureExtractor {
  protected readonly spectrogramCallback: SpectrogramCallback;
  readonly numFramesPerSpectrogram: number;
  readonly sampleRateHz: number;
  readonly fftSize: number;
  readonly columnTruncateLength: number;

  private stream: MediaStream;
  private audioContextConstructor: any;
  private audioContext: AudioContext;
  private analyser: AnalyserNode;

  private tracker: Tracker;

  private readonly ROTATING_BUFFER_SIZE_MULTIPLIER = 2;
  private freqData: Float32Array;
  private rotatingBufferNumFrames: number;
  private rotatingBuffer: Float32Array;

  private frameCount: number;
  private frameIntervalTask: any;

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

    this.spectrogramCallback = config.spectrogramCallback;
    this.numFramesPerSpectrogram = config.numFramesPerSpectrogram;
    this.sampleRateHz = config.sampleRateHz || 44100;
    this.fftSize = config.fftSize || 1024;
    this.columnTruncateLength = config.columnTruncateLength || this.fftSize;

    this.audioContextConstructor =
        (window as any).AudioContext || (window as any).webkitAudioContext;
  }

  async start(samples?: Float32Array): Promise<Float32Array[]|void> {
    this.stream =
        await navigator.mediaDevices.getUserMedia({audio: true, video: false});
    this.audioContext = this.audioContextConstructor() as AudioContext;
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
    const rotatingBufferSize = this.fftSize * this.rotatingBufferNumFrames;
    this.rotatingBuffer = new Float32Array(rotatingBufferSize);

    this.frameCount = 0;

    const overlapFactor = 0.5;  // TODO(cais): Get from config.
    this.tracker = new Tracker(
        Math.round(this.numFramesPerSpectrogram * (1 - overlapFactor)), 0);
    this.frameIntervalTask =
        setInterval(this.onAudioFrame, this.fftSize / this.sampleRateHz * 1e3);
  }

  private onAudioFrame() {
    this.analyser.getFloatFrequencyData(this.freqData);
    if (this.freqData[0] === -Infinity) {
      // No signal from audio input (microphone). Do nothing.
      return;
    }

    const freqDataSlice = this.freqData.slice(0, this.columnTruncateLength);
    const bufferPos = this.frameCount % this.rotatingBufferNumFrames;
    this.rotatingBuffer.set(
        freqDataSlice, bufferPos * this.columnTruncateLength);

    this.tracker.tick(true);
    if (this.tracker.shouldFire()) {
      const freqData = getFrequencyDataFromRotatingBuffer(
          this.rotatingBuffer, this.numFramesPerSpectrogram,
          this.columnTruncateLength,
          this.frameCount - this.numFramesPerSpectrogram);
      const inputTensor = getInputTensorFromFrequencyData(
          freqData, this.numFramesPerSpectrogram, this.columnTruncateLength);
      this.spectrogramCallback(inputTensor);
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

  setConfig(params: RecognizerConfigParams) {
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
  const size = freqData.length;
  const tensorBuffer = tf.buffer([size]);
  for (let i = 0; i < freqData.length; ++i) {
    tensorBuffer.set(freqData[i], i);
  }
  let output = tensorBuffer.toTensor().reshape([1, numFrames, fftLength, 1]);
  return toNormalize ? normalize(output) : output;
}

export class Tracker {
  readonly waitingPeriod: number;
  readonly refractoryPeriod: number;

  private counter: number;
  private state: number;
  private lastTriggerCounter: number;

  constructor(waitingPeriod: number, refactoryPeriod: number) {
    this.waitingPeriod = waitingPeriod;
    this.refractoryPeriod = refactoryPeriod;

    this.counter = 0;
    this.state = 0;
    this.lastTriggerCounter = -1;
  }

  // TODO(cais): What is trigger for anyway?
  tick(trigger: boolean) {
    if (this.state === 0) {
      if (trigger) {
        this.lastTriggerCounter = this.counter;
        this.state = 1;
      }
    } else if (this.state === 1) {
      if (this.counter - this.lastTriggerCounter === this.waitingPeriod) {
        this.state = 2;
      }
    } else if (this.state === 2) {
      if (this.refractoryPeriod === 0) {
        this.state = 0;
      } else {
        this.state = 3;
      }
    } else if (this.state === 3) {
      // In refractory period.
      if (this.counter - this.lastTriggerCounter >=
          this.waitingPeriod + 1 + this.refractoryPeriod) {
        this.state = 0;
      }
    }
    this.counter++;
  }

  shouldFire() {
    return this.state === 2;
  }

  isResting() {
    return this.state === 0;
  }
}