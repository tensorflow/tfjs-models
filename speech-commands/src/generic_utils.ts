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
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
export function concatenateArrayBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
  let totalByteLength = 0;
  buffers.forEach((buffer: ArrayBuffer) => {
    totalByteLength += buffer.byteLength;
  });

  const temp = new Uint8Array(totalByteLength);
  let offset = 0;
  buffers.forEach((buffer: ArrayBuffer) => {
    temp.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  });
  return temp.buffer;
}

/**
 * Concatenate Float32Arrays.
 *
 * @param xs Float32Arrays to concatenate.
 * @return The result of the concatenation.
 */
export function concatenateFloat32Arrays(xs: Float32Array[]): Float32Array {
  let totalLength = 0;
  xs.forEach(x => totalLength += x.length);
  const concatenated = new Float32Array(totalLength);
  let index = 0;
  xs.forEach(x => {
    concatenated.set(x, index);
    index += x.length;
  });
  return concatenated;
}

/** Encode a string as an ArrayBuffer. */
export function string2ArrayBuffer(str: string): ArrayBuffer {
  if (str == null) {
    throw new Error('Received null or undefind string');
  }
  // NOTE(cais): This implementation is inefficient in terms of memory.
  // But it works for UTF-8 strings. Just don't use on for very long strings.
  const strUTF8 = unescape(encodeURIComponent(str));
  const buf = new Uint8Array(strUTF8.length);
  for (let i = 0; i < strUTF8.length; ++i) {
    buf[i] = strUTF8.charCodeAt(i);
  }
  return buf.buffer;
}

/** Decode an ArrayBuffer as a string. */
export function arrayBuffer2String(buffer: ArrayBuffer): string {
  if (buffer == null) {
    throw new Error('Received null or undefind buffer');
  }
  const buf = new Uint8Array(buffer);
  return decodeURIComponent(escape(String.fromCharCode(...buf)));
}

/** Generate a pseudo-random UID. */
export function getUID(): string {
  function s4() {
    return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
  }
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() +
      s4() + s4();
}

export function getRandomInteger(min: number, max: number): number {
  return Math.floor((max - min) * Math.random()) + min;
}
