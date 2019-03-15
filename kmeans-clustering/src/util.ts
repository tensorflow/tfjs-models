import * as tf from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

function range(length: number): number[] {
  return Array.from(Array(length).keys());
}

function sampleWithoutReplacement(data: number[], nSamples: number): number[] {
  const dataCopy = data.slice(0);
  const output = [];
  for (let i = 0; i < nSamples; i++) {
    const randomIndex = Math.floor(Math.random() * dataCopy.length);
    output.push(dataCopy.splice(randomIndex, 1)[0]);
  }
  return output;
}

// run kmeans on random samples
export function kMeans(
  data: Tensor,
  nClusters: number,
  maxIterations = 300,
  tolerance = 10e-4
): Tensor {
  return tf.tidy(() => {
    let centroids = initCentroids(data, nClusters);
    let nearest;
    const samples = data.toFloat();
    for (let i = 0; i < maxIterations; i++) {
      nearest = assignToNearest(samples, centroids);
      // change updateCentroids schema
      centroids = updateCentroids(samples, nearest, nClusters);
    }
    // return an array of indices, representing the nearest "centroid" to each data point
    return nearest;
  });
}

/**
 * @param {Tensor} X - 2-D array of data points being clustered
 * @param {Number} nClusters - number of clusters
 * @return {Tensor} centroids
 */
export function initCentroids(X: Tensor, nClusters: number): Tensor {
  // select `nClusters` number of input data samples to be initial centroids
  const sampleIndices = sampleWithoutReplacement(range(X.shape[0]), nClusters);
  return tf.gather(X, sampleIndices);
}

/**
 * give each data instance a cluster assignment, based on which cluster centroid is nearest to it
 * @param {Tensor} samples - data points being clustered, shape = [nInstances, nDims]
 * @param {Tensor} centroids - positions of cluster centers, shape = [nClusters, nDims]
 * @param {Boolean} fillEmpty - whether to enforce all clusters to be non-empty
 * @returns {Tensor} indices of the nearest centroid to each data instance, shape = [nInscances]
 */
export function assignToNearest(samples: Tensor, centroids: Tensor): Tensor {
  return tf.tidy(() => {
    // perform outer operations by expanding dimensions
    const expandedVectors = tf.expandDims(samples, 0);
    const expandedCentroids = tf.expandDims(centroids, 1);
    const distances = tf.sum(
      tf.square(tf.sub(expandedVectors, expandedCentroids)),
      2
    );
    const mins = tf.argMin(distances, 0);
    return mins.toInt();
  });
}

/**
 * Calculate the average value of all instances in each cluster.
 * @param {Tensor} samples - data input, shape = [nInstances, nDims]
 * @param {Tensor} nearestIndices - indices of centroid to each data instance, shape = [nInstances, nClusters]
 * @param {Number} nClusters
 * @returns {Tensor} centroids of all instances in each cluster, shape = [nClusters, nDims]
 */
export function updateCentroids(
  samples: Tensor,
  nearestIndices: Tensor,
  nClusters: number
): Tensor {
  return tf.tidy(() => {
    const newCentroids = [];
    for (let i = 0; i < nClusters; i++) {
      const mask = tf.equal(nearestIndices, tf.scalar(i).toInt());
      const currentCentroid = tf.div(
        // set all masked instances to 0 by multiplying the mask tensor,
        // then sum across all instances
        tf.sum(tf.mul(tf.expandDims(mask.toFloat(), 1), samples), 0),
        // divided by number of instances
        tf.sum(mask.toFloat()),
      );
      newCentroids.push(currentCentroid);
    }
    return tf.stack(newCentroids);
  });
}

export function genRandomSamples(
  nClusters: number,
  nSamplesPerCluster: number,
  nFeatures = 2,
  embiggenFactor = 5,
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
        5,
        'float32',
        seed
      );

      const currentCentroid = tf.randomUniform(
        [1, nFeatures],
        embiggenFactor * -8,
        embiggenFactor * 8
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
