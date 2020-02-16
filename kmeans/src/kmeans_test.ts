import {
  describeWithFlags,
  NODE_ENVS,
} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as tf from '@tensorflow/tfjs-core';
import {kMeans} from './index';

describeWithFlags('KMeans', NODE_ENVS, () => {
  it('simple k-means', async () => {
    const X0 = tf.tensor([
      [0, 1],
      [1, 0],
      [0, 0.2],
      [10, 11],
      [11, 10],
      [10, 10.8],
      [100, 101],
      [101, 100],
      [100, 100.5],
    ]);
    const X1 = tf.tensor([
      [0.5, 0.5],
      [10.5, 10.5],
      [100.5, 100.5],
    ]);

    const model = kMeans({nClusters: 3, maxIter: 10});

    // Warm up
    await model.fitOneCycle(X0);
    const numTensorsBefore = tf.memory().numTensors;
    // console.log('--------- warm up finished');

    const result0 = await model.fitPredict(X0);
    // console.log('--------- fit finished');
    const result1 = await model.predict(X1);
    // console.log('--------- predict finished');

    for (let i = 0; i < 3; i += 1) {
      expect(result0[3 * i]).toEqual(result0[3 * i + 1]);
      expect(result0[3 * i]).toEqual(result1[i]);
    }
    expect(tf.memory().numTensors).toEqual(numTensorsBefore);
  });
});
