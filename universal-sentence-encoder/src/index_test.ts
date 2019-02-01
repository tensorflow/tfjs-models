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
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {load} from './index';

describeWithFlags('Universal Sentence Encoder', tf.test_util.NODE_ENVS, () => {
  beforeAll(() => {
    spyOn(tf, 'loadFrozenModel').and.callFake(() => {
      const model = {executeAsync: (inputs: string[]) => tf.zeros([1, 512])};
      return model;
    });
  });

  it('Universal Sentence Encoder doesn\'t leak', async () => {
    const model = await load();

    const numTensorsBefore = tf.memory().numTensors;

    await model.embed(['']);

    expect(tf.memory().numTensors).toBe(numTensorsBefore);
  });
});
