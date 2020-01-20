import * as tf from '@tensorflow/tfjs';
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as kmeans from './index';

describeWithFlags('kmeans', tf.test_util.NODE_ENVS, () => {
  it('simple nearest neighbors', async () => {
    const x0s = [
      tf.tensor1d([1, 1, 1, 1]), tf.tensor1d([1.1, .9, 1.2, .8]),
      tf.tensor1d([1.2, .8, 1.3, .7])
    ];
    const x1s = [
      tf.tensor1d([-1, -1, -1, -1]), tf.tensor1d([-1.1, -.9, -1.2, -.8]),
      tf.tensor1d([-1.2, -.8, -1.3, -.7])
    ];
    const classifier = kmeans.create();
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
    const classifier = kmeans.create();
    const x0 = tf.tensor1d([1.1, 1.1, 1.1, 1.1]);

    let errorMessage;
    try {
      await classifier.predictClass(x0);
    } catch (error) {
      errorMessage = error.message;
    }
    expect(errorMessage)
        .toMatch(/You have not added any examples to the KMeans/);
  });
});
