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
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {load} from './index';

describeWithFlags('MobileNet', tf.test_util.NODE_ENVS, () => {
  beforeAll(() => {
    spyOn(tf, 'loadModel').and.callFake(() => {
      const model = {
        predict: (x: tf.Tensor) => tf.zeros([x.shape[0], 1000]),
        layers: ['']
      };
      return model;
    });
  });

  it('MobileNet classify doesn\'t leak', async () => {
    const mobilenet = await load();

    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;

    const numTensorsBefore = tf.memory().numTensors;

    await mobilenet.classify(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore);
  });

  it('MobileNet infer doesn\'t leak', async () => {
    const mobilenet = await load();

    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;

    const numTensorsBefore = tf.memory().numTensors;

    mobilenet.infer(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore + 1);
  });
});
