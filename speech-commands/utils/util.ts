/**
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import * as tf from '@tensorflow/tfjs';

export function labelArrayToString(label: Float32Array, allLabels: string[]) {
  const [ind, ] = argmax(label);
  return allLabels[ind];
}

export function argmax(array: Float32Array) {
  let max = -Infinity;
  let argmax = -1;
  for (let i = 0; i < array.length; i++) {
    if (array[i] > max) {
      max = array[i];
      argmax = i;
    }
  }
  return [argmax, max];
}

export function getParameterByName(name: string, url?: string) {
  if (!url) url = window.location.href;
  name = name.replace(/[\[\]]/g, '\\$&');
  const regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
  if (!results) return null;
  if (!results[2]) return '';
  return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

export class Interval {
  private baseline: number;
  // tslint:disable-next-line:no-any
  private timer: any;
  constructor(private duration: number, private fn: Function) {
    this.baseline = undefined;
  }
  run() {
    if (this.baseline == null) {
      this.baseline = Date.now();
    }
    this.fn();
    const end = Date.now();
    this.baseline += this.duration;

    let nextTick = this.duration - (end - this.baseline);
    if (nextTick < 0) {
      nextTick = 0;
    }
    this.timer = setTimeout(this.run.bind(this), nextTick);
  }

  stop() {
    clearTimeout(this.timer);
  }
}

export function normalize(x: tf.Tensor) {
  return tf.tidy(() => {
    const mean = tf.mean(x);
    mean.print();
    const std = tf.sqrt(tf.mean(tf.square(tf.add(x, tf.neg(mean)))));
    return tf.div(tf.add(x, tf.neg(mean)), std);
  });
}

export function nextPowerOfTwo(value: number) {
  const exponent = Math.ceil(Math.log2(value));
  return 1 << exponent;
}

export function plotSpectrogram(
    canvas: HTMLCanvasElement, frequencyData: Float32Array[]) {
  // Get the maximum and minimum.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < frequencyData.length; ++i) {
    const x = frequencyData[i];
    for (let j = 1; j < x.length; ++j) {
      if (x[j] !== -Infinity) {
        if (x[j] < min) {
          min = x[j];
        }
        if (x[j] > max) {
          max = x[j];
        }
      }
    }
  }
  if (min >= max) {
    return;
  }

  const ctx = canvas.getContext('2d');
  const numTimeSteps = frequencyData.length;
  const pixelWidth = canvas.width / numTimeSteps;
  const pixelHeight = canvas.height / (frequencyData[0].length - 1);
  for (let i = 0; i < numTimeSteps; ++i) {
    const x = pixelWidth * i;
    const spectrum = frequencyData[i];
    if (spectrum[0] === -Infinity) {
      break;
    }
    for (let j = 1; j < frequencyData[0].length; ++j) {
      const y = canvas.height - (j + 1) * pixelHeight;

      let colorValue = (spectrum[j] - min) / (max - min);
      colorValue = Math.round(255 * colorValue);
      const fillStyle = `rgb(${colorValue},${colorValue},${colorValue})`;
      ctx.fillStyle = fillStyle;
      ctx.fillRect(x, y, pixelWidth, pixelHeight);
    }
  }
}

export function melSpectrogramToInput(spec: Float32Array[]): tf.Tensor {
  // Flatten this spectrogram into a 2D array.
  const times = spec.length;
  const freqs = spec[0].length;
  const data = new Float32Array(times * freqs);
  for (let i = 0; i < times; i++) {
    const mel = spec[i];
    const offset = i * freqs;
    data.set(mel, offset);
  }
  // Normalize the whole input to be in [0, 1].
  const shape: [number, number, number, number] = [1, times, freqs, 1];
  // this.normalizeInPlace(data, 0, 1);
  return tf.tensor4d(Array.prototype.slice.call(data), shape);
}
