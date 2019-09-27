import {Tensor, tensor, Tensor2D, Tensor3D, TensorContainer, util} from '@tensorflow/tfjs';
import {MicrophoneConfig} from '@tensorflow/tfjs-data/dist/types';

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
 * Testing Utilities for Browser Audio Feature Extraction.
 */

export class FakeAudioContext {
  readonly sampleRate = 44100;

  static createInstance() {
    return new FakeAudioContext();
  }

  createMediaStreamSource() {
    return new FakeMediaStreamAudioSourceNode();
  }

  createAnalyser() {
    return new FakeAnalyser();
  }

  close(): void {}
}

export class FakeAudioMediaStream {
  constructor() {}
  getTracks(): Array<{}> {
    return [];
  }
}

class FakeMediaStreamAudioSourceNode {
  constructor() {}
  connect(node: {}): void {}
}

class FakeAnalyser {
  fftSize: number;
  smoothingTimeConstant: number;
  private x: number;
  constructor() {
    this.x = 0;
  }

  getFloatFrequencyData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(this.x++);
    }
    data.set(new Float32Array(xs));
  }

  getFloatTimeDomainData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(-(this.x++));
    }
    data.set(new Float32Array(xs));
  }

  disconnect(): void {}
}

export class FakeMicrophoneIterator {
  fftSize: number;
  isClosed: boolean;
  includeSpectrogram: boolean;
  includeWaveform: boolean;
  freqData: Float32Array;
  timeData: Float32Array;
  numFrames: number;
  columnTruncateLength: number;
  sampleRateHz: 44100|48000;
  private x: number;

  constructor(microphoneConfig: MicrophoneConfig) {
    this.fftSize = microphoneConfig.fftSize || 1024;
    this.includeSpectrogram =
        microphoneConfig.includeSpectrogram === false ? false : true;
    this.includeWaveform =
        microphoneConfig.includeWaveform === true ? true : false;
    this.numFrames = microphoneConfig.numFramesPerSpectrogram;
    this.columnTruncateLength =
        microphoneConfig.columnTruncateLength || this.fftSize;
    this.numFrames = microphoneConfig.numFramesPerSpectrogram || 43;
    this.sampleRateHz = microphoneConfig.sampleRateHz;
    this.isClosed = false;
    this.freqData = new Float32Array(this.fftSize);
    this.timeData = new Float32Array(this.fftSize);

    this.x = 0;
  }

  async start(): Promise<void> {}

  async next(): Promise<IteratorResult<TensorContainer>> {
    if (this.isClosed) {
      return {value: null, done: true};
    }

    const freqDataQueue: Float32Array[] = [];
    const timeDataQueue: Float32Array[] = [];
    for (let i = 0; i < this.numFrames; i++) {
      if (this.includeSpectrogram) {
        this.getFloatFrequencyData(this.freqData);
        freqDataQueue.push(this.freqData.slice(0, this.columnTruncateLength));
      }
      if (this.includeWaveform) {
        this.getFloatTimeDomainData(this.timeData);
        timeDataQueue.push(this.timeData);
      }
    }
    let spectrogramTensor: Tensor;
    let waveformTensor: Tensor;
    if (this.includeSpectrogram) {
      const freqData = this.flattenQueue(freqDataQueue);
      spectrogramTensor = this.getTensorFromAudioDataArray(
          freqData, [this.numFrames, this.columnTruncateLength, 1]);
    }
    if (this.includeWaveform) {
      const timeData = this.flattenQueue(timeDataQueue);
      waveformTensor = this.getTensorFromAudioDataArray(
          timeData, [this.numFrames * this.fftSize, 1]);
    }
    return {
      value: {'spectrogram': spectrogramTensor, 'waveform': waveformTensor},
      done: false
    };
  }

  getFloatFrequencyData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(this.x++);
    }
    data.set(new Float32Array(xs));
  }

  getFloatTimeDomainData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(-(this.x++));
    }
    data.set(new Float32Array(xs));
  }

  async capture(): Promise<{spectrogram: Tensor3D, waveform: Tensor2D}> {
    return (await this.next()).value as
        {spectrogram: Tensor3D, waveform: Tensor2D};
  }

  stop() {
    this.isClosed = true;
  }

  private getTensorFromAudioDataArray(freqData: Float32Array, shape: number[]):
      Tensor {
    const vals = new Float32Array(util.sizeFromShape(shape));
    // If the data is less than the output shape, the rest is padded with zeros.
    vals.set(freqData, vals.length - freqData.length);
    return tensor(vals, shape);
  }

  // Return audio sampling rate in Hz
  getSampleRate(): number {
    return this.sampleRateHz;
  }

  private flattenQueue(queue: Float32Array[]): Float32Array {
    const frameSize = queue[0].length;
    const freqData = new Float32Array(queue.length * frameSize);
    queue.forEach((data, i) => freqData.set(data, i * frameSize));
    return freqData;
  }
}
