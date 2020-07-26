/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

/**
 * Rotates an image.
 *
 * @param image - Input tensor.
 * @param radians - Angle of rotation.
 * @param fillValue - The RGBA values to use in filling the leftover triangles
 * after rotation.
 * @param center - The center of rotation.
 */
export function rotate(
    image: tf.Tensor4D, radians: number, fillValue: number[]|number,
    center: [number, number]): tf.Tensor4D {
  const cpuBackend = tf.backend();

  const output = tf.buffer(image.shape, image.dtype);
  const [batch, imageHeight, imageWidth, numChannels] = image.shape;

  const centerX =
      imageWidth * (typeof center === 'number' ? center : center[0]);
  const centerY =
      imageHeight * (typeof center === 'number' ? center : center[1]);

  const sinFactor = Math.sin(radians);
  const cosFactor = Math.cos(radians);
  const imageVals = cpuBackend.readSync(image.dataId) as Float32Array;

  for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
    for (let row = 0; row < imageHeight; row++) {
      for (let col = 0; col < imageWidth; col++) {
        for (let channel = 0; channel < numChannels; channel++) {
          const coords = [batch, row, col, channel];

          const x = coords[2];
          const y = coords[1];

          let coordX = (x - centerX) * cosFactor - (y - centerY) * sinFactor;
          let coordY = (x - centerX) * sinFactor + (y - centerY) * cosFactor;

          coordX = Math.round(coordX + centerX);
          coordY = Math.round(coordY + centerY);

          let outputValue = fillValue;
          if (typeof fillValue !== 'number') {
            if (channel === 3) {
              outputValue = 255;
            } else {
              outputValue = fillValue[channel];
            }
          }

          if (coordX >= 0 && coordX < imageWidth && coordY >= 0 &&
              coordY < imageHeight) {
            const imageIdx = batchIdx * imageWidth * imageHeight * numChannels +
                coordY * (imageWidth * numChannels) + coordX * numChannels +
                channel;
            outputValue = imageVals[imageIdx];
          }

          const outIdx = batchIdx * imageWidth * imageHeight * numChannels +
              row * (imageWidth * numChannels) + col * numChannels + channel;
          output.values[outIdx] = outputValue as number;
        }
      }
    }
  }

  return output.toTensor() as tf.Tensor4D;
}
