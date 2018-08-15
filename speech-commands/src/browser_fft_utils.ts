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

// tslint:disable-next-line:no-any
export async function loadMetadataJson(url: string): Promise<any> {
  return await (await fetch(url)).json();
}

export function normalize(x: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const mean = tf.mean(x);
    const std = tf.sqrt(tf.mean(tf.square(tf.add(x, tf.neg(mean)))));
    return tf.div(tf.add(x, tf.neg(mean)), std);
  });
}

export function getAudioContextConstructor(): AudioContext {
  // tslint:disable-next-line:no-any
  return (window as any).AudioContext || (window as any).webkitAudioContext;
}

export async function getAudioMediaStream(): Promise<MediaStream> {
  return await navigator.mediaDevices.getUserMedia({audio: true, video: false});
}
