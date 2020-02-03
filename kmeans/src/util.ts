import * as tf from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

export function range(length: number): number[] {
  return Array.from(Array(length).keys());
}

export function sampleWithoutReplacement(
  data: number[],
  nSamples: number
): number[] {
  const dataCopy = data.slice(0);
  const output = [];
  for (let i = 0; i < nSamples; i++) {
    const randomIndex = Math.floor(Math.random() * dataCopy.length);
    output.push(dataCopy.splice(randomIndex, 1)[0]);
  }
  return output;
}

export function genRandomSamples(
  nClusters: number,
  nSamplesPerCluster: number,
  nFeatures = 2,
  variance = 1,
  embiggenFactor = 4,
  seed = 0
) {
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
