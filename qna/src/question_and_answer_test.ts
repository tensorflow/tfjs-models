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
import '@tensorflow/tfjs-backend-cpu';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';
import 'jasmine';

import {load} from './index';

describeWithFlags('qna', NODE_ENVS, () => {

  let model: tfconv.GraphModel;
  let executeSpy: jasmine.Spy<typeof model.execute>;
  beforeEach(() => {
    spyOn(tfconv, 'loadGraphModel').and.callFake((modelUrl: string) => {
      model = new tfconv.GraphModel(modelUrl);
      executeSpy = spyOn(model, 'execute')
          .and.callFake(
              (x: tf.Tensor) =>
                  [tf.tensor2d(
                       [
                         0, 0, 0, 0, 10, 20, 30, 20, 10, 0,
                         ...Array(374).fill(0)
                       ],
                       [1, 384]),
                   tf.tensor2d(
                       [
                         0, 0, 0, 0, 10, 20, 30, 20, 10, 20,
                         ...Array(374).fill(0)
                       ],
                       [1, 384])]);
      return Promise.resolve(model);
    });
  });

  it('qna detect method should not leak', async () => {
    const qna = await load();
    const numOfTensorsBefore = tf.memory().numTensors;

    await qna.findAnswers('question', 'context');

    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });

  it('qna detect method should generate output', async () => {
    const qna = await load();

    const data = await qna.findAnswers('question', 'context');

    expect(data).toEqual([]);
  });

  it('qna detect method should throw error if question is too long',
     async () => {
       const qna = await load();
       const question = 'question '.repeat(300);
       let result = undefined;
       try {
         result = await qna.findAnswers(question, 'context');
       } catch (error) {
         expect(error.message)
             .toEqual('The length of question token exceeds the limit (64).');
       }
       expect(result).toBeUndefined();
     });

  it('qna detect method should work for long context', async () => {
    const qna = await load();
    const context = 'text '.repeat(1000);
    executeSpy.and.returnValue(
      [tf.tensor2d(
        [
          0, 0, 0, 0, 10, 20, 30, 20, 10, 0,
          ...Array(384 * 6 - 10).fill(0)
        ],
        [6, 384]),
    tf.tensor2d(
        [
          0, 0, 0, 0, 10, 20, 30, 20, 10, 20,
          ...Array(384 * 6 - 10).fill(0)
        ],
        [6, 384])]
    );
    const data = await qna.findAnswers('question', context);
    expect(data.length).toEqual(5);
  });

  it('should allow custom model url', async () => {
    await load({modelUrl: 'https://google.com/model.json'});

    expect(tfconv.loadGraphModel)
        .toHaveBeenCalledWith(
            'https://google.com/model.json', {fromTFHub: false});
  });

  it('should populate the startIndex and endIndex', async () => {
    const qna = await load();

    const result = await qna.findAnswers('question', 'this is answer for you!');

    expect(result).toEqual([
      {text: 'answer', score: 60, startIndex: 8, endIndex: 14},
      {text: 'answer for', score: 50, startIndex: 8, endIndex: 18},
      {text: 'answer for you!', score: 50, startIndex: 8, endIndex: 23},
      {text: 'is answer', score: 50, startIndex: 5, endIndex: 14},
      {text: 'is', score: 40, startIndex: 5, endIndex: 7}
    ]);
  });
});
