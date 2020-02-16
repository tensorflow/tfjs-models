import * as tf from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

export function range(length: number): number[] {
  return Array.from(Array(length).keys());
}

export async function sampleWithoutReplacement(
  data: number[],
  nSamples: number,
  seed = 0
): Promise<number[]> {
  // Fisher-Yates sample without replacement
  const dataCopy = data.slice(0);
  for (let i = 0; i < nSamples; i++) {
    const randomIndexArr = await tf
      .randomUniform([1], i, dataCopy.length, 'int32', seed + i)
      .data();

    const randomIndex = randomIndexArr[0];
    const sampled = dataCopy[randomIndex];
    dataCopy[randomIndex] = dataCopy[i];
    dataCopy[i] = sampled;
  }
  return dataCopy.slice(0, nSamples);
}

export function genRandomSamples(
  nClusters: number,
  nSamplesPerCluster: number,
  nFeatures = 2,
  variance = 1,
  embiggenFactor = 4,
  seed = 0
): {centroids: Tensor; samples: Tensor} {
  return tf.tidy(() => {
    const slices = [];
    let centroidsArr: Tensor[] = [];
    // Create samples for each cluster
    for (let i = 0; i < nClusters; i++) {
      let samples = tf.randomNormal(
        [nSamplesPerCluster, nFeatures],
        0,
        variance,
        'float32',
        seed
      );

      const currentCentroid = tf.randomUniform(
        [1, nFeatures],
        embiggenFactor * -1,
        embiggenFactor
      );
      samples = samples.add(currentCentroid);
      centroidsArr = centroidsArr.concat(currentCentroid);
      slices.push(samples);
    }
    // Create a big "samples" dataset
    const samples = tf.concat(slices, 0);
    const centroids = tf.concat(centroidsArr, 0);
    return {
      centroids,
      samples,
    };
  });
}
