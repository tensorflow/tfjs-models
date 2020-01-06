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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {expectArrayBuffersEqual} from '@tensorflow/tfjs-core/dist/test_util';

import {load} from './index';

describeWithFlags('mobileBert', NODE_ENVS, () => {
  beforeEach(() => {
    spyOn(tfconv, 'loadGraphModel').and.callFake(() => {
      const model = {
        execute: (
            x: tf
                .Tensor) => [tf.ones([1, 1917, 90]), tf.ones([1, 1917, 1, 4])]
      };
      return model;
    });
  });

  it('mobileBert detect method should not leak', async () => {
    const mobileBert = await load();
    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;
    const numOfTensorsBefore = tf.memory().numTensors;

    await mobileBert.findAnswers('question', 'context');

    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });

  it('mobileBert detect method should generate output', async () => {
    const mobileBert = await load();
    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;

    const data = mobileBert.findAnswers('question', 'context');

    expect(data).toEqual([]);
  });

  it('should allow custom model url', async () => {
    const mobileBert = await load({modelUrl: 'https://test.org/model.json'});

    expect(tfconv.loadGraphModel)
        .toHaveBeenCalledWith('https://test.org/model.json');
  });
});
