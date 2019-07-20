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
import {describeWithFlags, NODE_ENVS,} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {readFileSync} from 'fs';
import {decode} from 'jpeg-js';
import {resolve} from 'path';

import {load} from '.';
import {Point} from './geometry';

describeWithFlags('TextDetection', NODE_ENVS, () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 40000;
  });
  it('Text detection does not leak.', async () => {
    const model = await load();

    const empty = tf.zeros([227, 227, 3]) as tf.Tensor3D;
    const meaningful = tf.tidy(() => {
      const testImage =
          decode(readFileSync(resolve(__dirname, 'assets/example.jpeg')), true);
      const rawData = tf.tensor(testImage.data, [
                          testImage.height, testImage.width, 4
                        ]).arraySync() as number[][][];
      const inputBuffer =
          tf.buffer([testImage.height, testImage.width, 3], 'int32');
      for (let columnIndex = 0; columnIndex < testImage.height; ++columnIndex) {
        for (let rowIndex = 0; rowIndex < testImage.width; ++rowIndex) {
          for (let channel = 0; channel < 3; ++channel) {
            inputBuffer.set(
                rawData[columnIndex][rowIndex][channel], columnIndex, rowIndex,
                channel);
          }
        }
      }

      return inputBuffer.toTensor();
    }) as tf.Tensor3D;
    const numOfTensorsBefore = tf.memory().numTensors;

    await model.predict(empty);
    await model.predict(meaningful);

    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });
  it('TextDetection produces sensible results.', async () => {
    const model = await load();

    const input = tf.tidy(() => {
      const testImage =
          decode(readFileSync(resolve(__dirname, 'assets/example.jpeg')), true);
      const rawData = tf.tensor(testImage.data, [
                          testImage.height, testImage.width, 4
                        ]).arraySync() as number[][][];
      const inputBuffer =
          tf.buffer([testImage.height, testImage.width, 3], 'int32');
      for (let columnIndex = 0; columnIndex < testImage.height; ++columnIndex) {
        for (let rowIndex = 0; rowIndex < testImage.width; ++rowIndex) {
          for (let channel = 0; channel < 3; ++channel) {
            inputBuffer.set(
                rawData[columnIndex][rowIndex][channel], columnIndex, rowIndex,
                channel);
          }
        }
      }

      return inputBuffer.toTensor();
    }) as tf.Tensor3D;
    const boxes = await model.predict(input);
    const xCoords = new Set<number>();
    const yCoords = new Set<number>();
    const points = new Set<Point>();
    let conds = [
      boxes.length === 1,
    ];
    if (boxes.length > 0) {
      for (const box of boxes) {
        const [upright, upleft, downleft, downright] = box;
        const left = (upleft.sub(downleft)).norm();
        const right = (upright.sub(downright)).norm();
        const up = (upright.sub(upleft)).norm();
        const down = (downright.sub(downleft)).norm();
        const epsilon = 1e-9;
        conds = conds.concat([
          left > 0, right > 0, up > 0, down > 0, (right - left) < epsilon,
          (up - down) < epsilon, (right * up) > epsilon
        ]);
        for (const point of box) {
          xCoords.add(Math.round(point.x));
          yCoords.add(Math.round(point.y));
          points.add(point);
        }
      }
    }
    conds = conds.concat([
      xCoords.size === 2,
      yCoords.size === 2,
      points.size === 4,
    ]);
    expect(conds).toEqual(Array.from(new Array(conds.length), () => true));
  });
});
