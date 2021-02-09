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

import * as tf from '@tensorflow/tfjs-core';
import {promisify} from 'util';

import {RawAudioData} from './types';

export async function loadMetadataJson(url: string):
    Promise<{wordLabels: string[]}> {
  const HTTP_SCHEME = 'http://';
  const HTTPS_SCHEME = 'https://';
  const FILE_SCHEME = 'file://';
  if (url.indexOf(HTTP_SCHEME) === 0 || url.indexOf(HTTPS_SCHEME) === 0) {
    const response = await fetch(url);
    const parsed = await response.json();
    return parsed;
  } else if (url.indexOf(FILE_SCHEME) === 0) {
    // tslint:disable-next-line:no-require-imports
    const fs = require('fs');
    const readFile = promisify(fs.readFile);

    return JSON.parse(
        await readFile(url.slice(FILE_SCHEME.length), {encoding: 'utf-8'}));
  } else {
    throw new Error(
        `Unsupported URL scheme in metadata URL: ${url}. ` +
        `Supported schemes are: http://, https://, and ` +
        `(node.js-only) file://`);
  }
}

let EPSILON: number = null;

/**
 * Normalize the input into zero mean and unit standard deviation.
 *
 * This function is safe against divison-by-zero: In case the standard
 * deviation is zero, the output will be all-zero.
 *
 * @param x Input tensor.
 * @param y Output normalized tensor.
 */
export function normalize(x: tf.Tensor): tf.Tensor {
  if (EPSILON == null) {
    EPSILON = tf.backend().epsilon();
  }
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(x);
    // Add an EPSILON to the denominator to prevent division-by-zero.
    return tf.div(tf.sub(x, mean), tf.add(tf.sqrt(variance), EPSILON));
  });
}

/**
 * Z-Normalize the elements of a Float32Array.
 *
 * Subtract the mean and divide the result by the standard deviation.
 *
 * @param x The Float32Array to normalize.
 * @return Noramlzied Float32Array.
 */
export function normalizeFloat32Array(x: Float32Array): Float32Array {
  if (x.length < 2) {
    throw new Error(
        'Cannot normalize a Float32Array with fewer than 2 elements.');
  }
  if (EPSILON == null) {
    EPSILON = tf.backend().epsilon();
  }
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(tf.tensor1d(x));
    const meanVal = mean.arraySync() as number;
    const stdVal = Math.sqrt(variance.arraySync() as number);
    const yArray = Array.from(x).map(y => (y - meanVal) / (stdVal + EPSILON));
    return new Float32Array(yArray);
  });
}

export function getAudioContextConstructor(): AudioContext {
  // tslint:disable-next-line:no-any
  return (window as any).AudioContext || (window as any).webkitAudioContext;
}

export async function getAudioMediaStream(
    audioTrackConstraints?: MediaTrackConstraints): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: audioTrackConstraints == null ? true : audioTrackConstraints,
    video: false
  });
}

/**
 * Play raw audio waveform
 * @param rawAudio Raw audio data, including the waveform and the sampling rate.
 * @param onEnded Callback function to execute when the playing ends.
 */
export function playRawAudio(
    rawAudio: RawAudioData, onEnded: () => void|Promise<void>): void {
  const audioContextConstructor =
      // tslint:disable-next-line:no-any
      (window as any).AudioContext || (window as any).webkitAudioContext;
  const audioContext: AudioContext = new audioContextConstructor();
  const arrayBuffer =
      audioContext.createBuffer(1, rawAudio.data.length, rawAudio.sampleRateHz);
  const nowBuffering = arrayBuffer.getChannelData(0);
  nowBuffering.set(rawAudio.data);
  const source = audioContext.createBufferSource();
  source.buffer = arrayBuffer;
  source.connect(audioContext.destination);
  source.start();
  source.onended = () => {
    if (onEnded != null) {
      onEnded();
    }
  };
}
