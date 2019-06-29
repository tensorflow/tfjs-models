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

import * as tf from '@tensorflow/tfjs';
import {
  describeWithFlags,
  NODE_ENVS,
} from '@tensorflow/tfjs-core/dist/jasmine_util';
import { readFileSync } from 'fs';
import { EfficientNet } from '.';
import { decode } from 'jpeg-js';
import { resolve } from 'path';
import { EfficientNetBaseModel } from './types';
import { config } from './config';

const areEqualNumbers = (firstArray: number[], secondArray: number[]) => {
  if (firstArray.length !== secondArray.length) {
    return false;
  }
  for (let idx = 0; idx < firstArray.length; idx++) {
    if (firstArray[idx] !== secondArray[idx]) {
      return false;
    }
  }
  return true;
};

describeWithFlags('EfficientNet', NODE_ENVS, () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 40000;
  });

  it('EfficientNet preprocessing produces matching dimensions', async () => {
    const input = tf.zeros([314, 529, 3]) as tf.Tensor3D;
    const models = Array.from(Array(6).keys()).map(
      idx => `b${idx}` as EfficientNetBaseModel
    );
    const isMatching = Array(6).fill(false);
    models.forEach((model, idx) => {
      const targetSize = config.CROP_SIZE[model];
      isMatching[idx] = areEqualNumbers(
        EfficientNet.preprocess(model, input).shape,
        [1, targetSize, targetSize, 3]
      );
    });
    expect(isMatching).toEqual(Array(6).fill(true));
  });

  it('EfficientNet predictions do not leak.', async () => {
    const model = new EfficientNet('b0');

    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;
    const numOfTensorsBefore = tf.memory().numTensors;

    await model.predict(x, 10);
    await model.dispose();

    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });

  it('EfficientNet produces sensible results.', async () => {
    const model = new EfficientNet('b0');

    const input = tf.tidy(() => {
      const testImage = decode(
        readFileSync(resolve(__dirname, 'input_test.jpg')),
        true
      );
      const rawData = tf
        .tensor(testImage.data, [testImage.height, testImage.width, 4])
        .arraySync() as number[][][];
      const inputBuffer = tf.buffer(
        [testImage.height, testImage.width, 3],
        'int32'
      );
      for (let columnIndex = 0; columnIndex < testImage.height; ++columnIndex) {
        for (let rowIndex = 0; rowIndex < testImage.width; ++rowIndex) {
          for (let channel = 0; channel < 3; ++channel) {
            inputBuffer.set(
              rawData[columnIndex][rowIndex][channel],
              columnIndex,
              rowIndex,
              channel
            );
          }
        }
      }

      return inputBuffer.toTensor();
    }) as tf.Tensor3D;
    const predictions = await model.predict(input, 10);
    const isPanda = predictions[0].className.includes('panda');
    await model.dispose();
    expect(isPanda).toEqual(true);
  });
});
