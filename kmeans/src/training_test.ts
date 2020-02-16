import {
  describeWithFlags,
  NODE_ENVS,
} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {tensor} from '@tensorflow/tfjs-core';
import {initCentroids, assignToNearest, updateCentroids} from './training';

describeWithFlags('KMeans', NODE_ENVS, () => {
  it('initCentroids', () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    expect(initCentroids(X, 2).shape).toEqual([2, 3]);
  });

  it('updateCentroids', () => {
    const samples1 = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 1]);
    const indices1 = tensor([0, 0, 1, 2, 1, 2, 1, 2, 1, 2]);
    const centroids1 = updateCentroids(samples1, indices1, 3);

    expect(centroids1.shape).toEqual([3, 1]);
    expect(Array.from(centroids1.dataSync())).toEqual([1.5, 6, 7]);
  });

  it('assignToNearest', () => {
    const samples1 = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 1]);
    const centroids1 = tensor([0, 5.5, 10], [3, 1]);
    const nearest1 = assignToNearest(samples1, centroids1);

    expect(nearest1.shape).toEqual([10]);
    const nearest1Arr = Array.from(nearest1.dataSync());
    expect(nearest1Arr).toEqual([0, 0, 1, 1, 1, 1, 1, 2, 2, 2]);
  });
});
