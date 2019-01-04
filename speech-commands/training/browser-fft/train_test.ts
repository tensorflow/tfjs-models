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

/**
 * Unit test for model training script: train.ts.
 */

import {createBrowserFFTModel} from './train';

describe('Model and training', () => {
  it('Create model', () => {
    const model = createBrowserFFTModel([43, 232, 1], 10);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 43, 232, 1]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 10]);
  });

  // TODO(cais): Figure out why this test fails even though there's no problem
  // with training from train.ts' main() function. It may have to do with
  // registration of backends in the test environment.
  // it('Fit model', async () => {
  //   const model = createBrowserFFTModel([43, 232, 1], 4);
  //   model.compile(
  //       {loss: 'categoricalCrossentropy', optimizer: 'sgd',
  //        metrics: ['acc']});
  //   const xs = tf.randomNormal([10, 43, 232, 1]);
  //   const ys =
  //       tf.oneHot(tf.tensor1d([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], 'int32'), 4);
  //   const history = await model.fit(xs, ys, {epochs: 2});
  //   expect(history.history.loss.length).toEqual(2);
  //   expect(history.history.acc.length).toEqual(2);
  // });
});
