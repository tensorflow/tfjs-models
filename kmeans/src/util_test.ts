import {
  describeWithFlags,
  NODE_ENVS,
} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as tf from '@tensorflow/tfjs-core';
import {range, sampleWithoutReplacement} from './util';

describeWithFlags('KMeans', NODE_ENVS, () => {
  it('sampleWithoutReplacement', async () => {
    const numTensorsBefore = tf.memory().numTensors;
    const nSamples = 3;
    const sampleIndices = await sampleWithoutReplacement(range(100), nSamples);
    const samplesSet = new Set(sampleIndices);
    // test all samples are different
    expect(samplesSet.size).toBe(nSamples);
    expect(tf.memory().numTensors).toEqual(numTensorsBefore);
  });
});
