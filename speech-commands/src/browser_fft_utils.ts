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

export async function loadMetadataJson(url: string):
    Promise<{words: string[]}> {
  const HTTP_SCHEME = 'http://';
  const HTTPS_SCHEME = 'https://';
  const FILE_SCHEME = 'file://';
  if (url.indexOf(HTTP_SCHEME) === 0 || url.indexOf(HTTPS_SCHEME) === 0) {
    return await (await fetch(url)).json();
  } else if (url.indexOf(FILE_SCHEME) === 0) {
    // tslint:disable-next-line:no-require-imports
    const fs = require('fs');
    const content = JSON.parse(
        fs.readFileSync(url.slice(FILE_SCHEME.length), {encoding: 'utf-8'}));
    return content;
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
    EPSILON = tf.ENV.get('EPSILON');
  }
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(x);
    // Add an EPSILON to the denominator to prevent division-by-zero.
    return x.sub(mean).div(variance.sqrt().add(EPSILON));
  });
}

export function getAudioContextConstructor(): AudioContext {
  // tslint:disable-next-line:no-any
  return (window as any).AudioContext || (window as any).webkitAudioContext;
}

export async function getAudioMediaStream(): Promise<MediaStream> {
  return await navigator.mediaDevices.getUserMedia({audio: true, video: false});
}
