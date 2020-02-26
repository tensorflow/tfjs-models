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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {load} from './index';

describeWithFlags('MobileNet', NODE_ENVS, () => {
  beforeAll(() => {
    spyOn(tfconv, 'loadGraphModel').and.callFake(() => {
      const model = {
        predict: (x: tf.Tensor) => tf.zeros([x.shape[0], 1001]),
        execute: (x: tf.Tensor, nodeName: string) =>
            tf.zeros([x.shape[0], 1, 1, 1024]),
      };
      return model;
    });
  });

  it('batched input logits', async () => {
    const mobilenet = await load();
    const img: tf.Tensor4D = tf.zeros([3, 227, 227, 3]);
    const logits = mobilenet.infer(img);
    expect(logits.shape).toEqual([3, 1000]);
  });

  it('batched input embeddings', async () => {
    const mobilenet = await load();
    const img: tf.Tensor4D = tf.zeros([3, 227, 227, 3]);
    const embedding = mobilenet.infer(img, true /* embedding */);
    expect(embedding.shape).toEqual([3, 1024]);
  });

  it('MobileNet classify doesn\'t leak', async () => {
    const mobilenet = await load();
    const x: tf.Tensor3D = tf.zeros([227, 227, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    await mobilenet.classify(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore);
  });

  it('MobileNet infer doesn\'t leak', async () => {
    const mobilenet = await load();
    const x = tf.zeros([227, 227, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    mobilenet.infer(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore + 1);
  });
});
