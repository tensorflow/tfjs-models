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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {load} from './index';

let model;
describeWithFlags('mobileBert', NODE_ENVS, () => {
  beforeEach(() => {
    spyOn(tfconv, 'loadGraphModel').and.callFake((modelUrl: string) => {
      model = new tfconv.GraphModel(modelUrl);
      spyOn(model, 'execute')
          .and.callFake((x: tf.Tensor) => [tf.ones([1, 10]), tf.ones([1, 10])]);
      return Promise.resolve(model);
    });
  });

  it('mobileBert detect method should not leak', async () => {
    const mobileBert = await load();
    const numOfTensorsBefore = tf.memory().numTensors;

    await mobileBert.findAnswers('question', 'context');

    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });

  it('mobileBert detect method should generate output', async () => {
    const mobileBert = await load();

    const data = await mobileBert.findAnswers('question', 'context');

    expect(data).toEqual([]);
  });

  it('mobileBert detect method should throw error if question is too long',
     async () => {
       const mobileBert = await load();
       const question = 'question '.repeat(300);
       let result = undefined;
       try {
         result = await mobileBert.findAnswers(question, 'context');
       } catch (error) {
         expect(error.message)
             .toEqual('The length of question token exceeds the limit (64).');
       }
       expect(result).toBeUndefined();
     });

  it('mobileBert detect method should work for long context', async () => {
    const mobileBert = await load();
    const context = 'text '.repeat(1000);

    const data = await mobileBert.findAnswers('question', context);

    expect(data.length).toEqual(5);
  });

  it('should allow custom model url', async () => {
    await load({modelUrl: 'https://google.com/model.json'});

    expect(tfconv.loadGraphModel)
        .toHaveBeenCalledWith(
            'https://google.com/model.json', {fromTFHub: false});
  });

  it('should populate the startIndex and endIndex', async () => {
    const mobileBert = await load();
    model.execute.and.callFake(
        (x: tf.Tensor) =>
            [tf.tensor2d([0, 0, 0, 0, 1, 2, 3, 2, 1, 0], [1, 10]),
             tf.tensor2d([0, 0, 0, 0, 1, 2, 3, 2, 1, 0], [1, 10])]);

    const result =
        await mobileBert.findAnswers('question', 'this is the answer for you!');

    expect(result).toEqual([
      {text: 'answer', score: 6, startIndex: 12, endIndex: 18},
      {text: 'answer for', score: 5, startIndex: 12, endIndex: 22},
      {text: 'the answer', score: 5, startIndex: 8, endIndex: 18},
      {text: 'answer for you', score: 4, startIndex: 12, endIndex: 25},
      {text: 'the', score: 4, startIndex: 8, endIndex: 11}
    ]);
  });
});
