/**
 * Copyright 2019 Google LLC
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

/**
 * Save Float32Array in arbitrarily sized chunks.
 * Load Float32Array in arbitrarily sized chunks.
 * Determine if there's enough data to grab a certain amount.
 */
export class CircularAudioBuffer {
  buffer: Float32Array;
  // The index that we are currently full up to. New data is written from
  // [currentIndex + 1, maxLength]. Data can be read from [0, currentIndex].
  currentIndex: number;

  constructor(maxLength: number) {
    this.buffer = new Float32Array(maxLength);
    this.currentIndex = 0;
  }

  /**
   * Add a new buffer of data. Called when we get new audio input samples.
   */
  addBuffer(newBuffer: Float32Array) {
    // Do we have enough data in this buffer?
    const remaining = this.buffer.length - this.currentIndex;
    if (this.currentIndex + newBuffer.length > this.buffer.length) {
      console.error(
          `Not enough space to write ${newBuffer.length}` +
          ` to this circular buffer with ${remaining} left.`);
      return;
    }
    this.buffer.set(newBuffer, this.currentIndex);
    this.currentIndex += newBuffer.length;
  }

  /**
   * How many samples are stored currently?
   */
  getLength() {
    return this.currentIndex;
  }

  /**
   * How much space remains?
   */
  getRemainingLength() {
    return this.buffer.length - this.currentIndex;
  }

  /**
   * Return the first N samples of the buffer, and remove them. Called when we
   * want to get a buffer of audio data of a fixed size.
   */
  popBuffer(length: number) {
    // Do we have enough data to read back?
    if (this.currentIndex < length) {
      console.error(
          `This circular buffer doesn't have ${length} entries in it.`);
      return undefined;
    }
    if (length === 0) {
      console.warn(`Calling popBuffer(0) does nothing.`);
      return undefined;
    }
    const popped = this.buffer.slice(0, length);
    const remaining = this.buffer.slice(length, this.buffer.length);
    // Remove the popped entries from the buffer.
    this.buffer.fill(0);
    this.buffer.set(remaining, 0);
    // Send the currentIndex back.
    this.currentIndex -= length;
    return popped;
  }

  /**
   * Get the the first part of the buffer without mutating it.
   */
  getBuffer(length?: number) {
    if (!length) {
      length = this.getLength();
    }
    // Do we have enough data to read back?
    if (this.currentIndex < length) {
      console.error(
          `This circular buffer doesn't have ${length} entries in it.`);
      return undefined;
    }
    return this.buffer.slice(0, length);
  }

  clear() {
    this.currentIndex = 0;
    this.buffer.fill(0);
  }
}
