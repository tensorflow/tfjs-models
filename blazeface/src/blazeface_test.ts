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

import {BlazeFaceModel} from './face';
import * as blazeface from './index';

describeWithFlags('BlazeFace', NODE_ENVS, () => {
  let model: BlazeFaceModel;
  beforeAll(async () => {
    spyOn(tfconv, 'loadGraphModel')
        .and.callFake(() => ({predict: () => tf.zeros([1, 896, 17])}));

    model = await blazeface.load();
  });

  it('estimateFaces does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([224, 224, 3]);
    const beforeTensors = tf.memory().numTensors;
    await model.estimateFaces(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });
});
