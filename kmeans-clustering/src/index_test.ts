/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as knnClassier from './index';

describeWithFlags('KNNClassifier', tf.test_util.NODE_ENVS, () => {
  it('simple nearest neighbors', async () => {
    const x0s = [
      tf.tensor1d([1, 1, 1, 1]), tf.tensor1d([1.1, .9, 1.2, .8]),
      tf.tensor1d([1.2, .8, 1.3, .7])
    ];
    const x1s = [
      tf.tensor1d([-1, -1, -1, -1]), tf.tensor1d([-1.1, -.9, -1.2, -.8]),
      tf.tensor1d([-1.2, -.8, -1.3, -.7])
    ];
    const classifier = knnClassier.create();
    x0s.forEach(x0 => classifier.addExample(x0, 0));
    x1s.forEach(x1 => classifier.addExample(x1, 1));

    const x0 = tf.tensor1d([1.1, 1.1, 1.1, 1.1]);
    const x1 = tf.tensor1d([-1.1, -1.1, -1.1, -1.1]);

    // Warmup.
    await classifier.predictClass(x0);

    const numTensorsBefore = tf.memory().numTensors;

    const result0 = await classifier.predictClass(x0);
    expect(result0.classIndex).toBe(0);

    const result1 = await classifier.predictClass(x1);
    expect(result1.classIndex).toBe(1);

    expect(tf.memory().numTensors).toEqual(numTensorsBefore);
  });

  it('calling predictClass before adding example throws', async () => {
    const classifier = knnClassier.create();
    const x0 = tf.tensor1d([1.1, 1.1, 1.1, 1.1]);

    let errorMessage;
    try {
      await classifier.predictClass(x0);
    } catch (error) {
      errorMessage = error.message;
    }
    expect(errorMessage)
        .toMatch(/You have not added any exaples to the KNN classifier/);
  });
});
