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
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as knnClassifier from './index';

describeWithFlags('KNNClassifier', NODE_ENVS, () => {
  it('simple nearest neighbors', async () => {
    const x0s = [
      tf.tensor1d([1, 1, 1, 1]), tf.tensor1d([1.1, 0.9, 1.2, 0.8]),
      tf.tensor1d([1.2, 0.8, 1.3, 0.7])
    ];
    const x1s = [
      tf.tensor1d([-1, -1, -1, -1]), tf.tensor1d([-1.1, -0.9, -1.2, -0.8]),
      tf.tensor1d([-1.2, -0.8, -1.3, -0.7])
    ];
    const classifier = knnClassifier.create();
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

    classifier.dispose();
  });

  it('calling predictClass before adding example throws', async () => {
    const classifier = knnClassifier.create();
    const x0 = tf.tensor1d([1.1, 1.1, 1.1, 1.1]);

    let errorMessage;
    try {
      await classifier.predictClass(x0);
    } catch (error) {
      errorMessage = error.message;
    }
    expect(errorMessage)
        .toMatch(/You have not added any examples to the KNN classifier/);

    classifier.dispose();
  });

  it('examples with classId that does not start at 0 works', async () => {
    const classifier = knnClassifier.create();
    classifier.addExample(tf.tensor2d([5, 2], [2, 1]), 1);
    classifier.addExample(tf.tensor2d([6, 1], [2, 1]), 2);
    const result = await classifier.predictClass(tf.tensor2d([3, 3], [2, 1]));
    expect(result.classIndex).toBe(0);
    expect(result.label).toBe('1');
    expect(result.confidences).toEqual({'1': 0.5, '2': 0.5});
    expect(classifier.getClassExampleCount()).toEqual({1: 1, 2: 1});

    classifier.dispose();
  });

  it('examples with classId 5, 7 and 9', async () => {
    const classifier = knnClassifier.create();
    classifier.addExample(tf.tensor1d([7, 7]), 7);
    classifier.addExample(tf.tensor1d([5, 5]), 5);
    classifier.addExample(tf.tensor1d([9, 9]), 9);
    classifier.addExample(tf.tensor1d([5, 5]), 5);
    const result = await classifier.predictClass(tf.tensor1d([5, 5]));
    expect(result.classIndex).toBe(1);
    expect(result.label).toBe('5');
    expect(result.confidences).toEqual({5: 2 / 3, 7: 1 / 3, 9: 0});
    expect(classifier.getClassExampleCount()).toEqual({5: 2, 7: 1, 9: 1});

    classifier.dispose();
  });

  it('examples with string labels', async () => {
    const classifier = knnClassifier.create();
    classifier.addExample(tf.tensor1d([7, 7]), 'a');
    classifier.addExample(tf.tensor1d([5, 5]), 'b');
    classifier.addExample(tf.tensor1d([9, 9]), 'c');
    classifier.addExample(tf.tensor1d([5, 5]), 'b');
    const result = await classifier.predictClass(tf.tensor1d([5, 5]));
    expect(result.classIndex).toBe(1);
    expect(result.label).toBe('b');
    expect(result.confidences).toEqual({b: 2 / 3, a: 1 / 3, c: 0});
    expect(classifier.getClassExampleCount()).toEqual({b: 2, a: 1, c: 1});

    classifier.dispose();
  });

  it('getClassifierDataset', () => {
    const classifier = knnClassifier.create();
    classifier.addExample(tf.tensor1d([5, 5.1]), 5);
    classifier.addExample(tf.tensor1d([7, 7]), 7);
    classifier.addExample(tf.tensor1d([5.2, 5.3]), 5);
    classifier.addExample(tf.tensor1d([9, 9]), 9);

    const dataset = classifier.getClassifierDataset();
    expect(Object.keys(dataset)).toEqual(['5', '7', '9']);
    expect(dataset[5].shape).toEqual([2, 2]);
    expect(dataset[7].shape).toEqual([1, 2]);
    expect(dataset[9].shape).toEqual([1, 2]);

    classifier.dispose();
  });

  it('clearClass', async () => {
    expect(tf.memory().numTensors).toBe(0);
    const classifier = knnClassifier.create();
    tf.tidy(() => {
      classifier.addExample(tf.tensor1d([5, 5]), 5);
      classifier.addExample(tf.tensor1d([7, 7]), 7);
      classifier.addExample(tf.tensor1d([5, 5]), 5);
      classifier.addExample(tf.tensor1d([9, 9]), 9);
    });
    const numTensorsBefore = tf.memory().numTensors;

    expect(classifier.getClassExampleCount()).toEqual({5: 2, 7: 1, 9: 1});
    expect(classifier.getNumClasses()).toBe(3);
    expect(numTensorsBefore).toBe(3);

    classifier.clearClass(5);
    expect(classifier.getClassExampleCount()).toEqual({7: 1, 9: 1});
    expect(classifier.getNumClasses()).toBe(2);
    const numTensorsAfter = tf.memory().numTensors;
    expect(numTensorsAfter).toBe(2);

    classifier.clearAllClasses();
    expect(classifier.getClassExampleCount()).toEqual({});
    expect(classifier.getNumClasses()).toBe(0);
    expect(tf.memory().numTensors).toBe(0);

    classifier.dispose();
  });
});
