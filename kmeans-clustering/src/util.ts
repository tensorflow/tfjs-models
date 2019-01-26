import * as tf from '@tensorflow/tfjs';

// run kmeans on random samples
export const kMeans = (data: number[][], nClusters: number, maxIterations = 300, tolerance = 10e-4):
  tf.Tensor => {

  return tf.tidy(() => {
    let centroids = initCentroids(data, nClusters);
    let nearest;
    const samples = tf.tensor2d(data).toFloat();
    for (let i = 0; i < maxIterations; i++) {
      nearest = assignToNearest(samples, centroids);
      centroids = updateCentroids(samples, nearest, nClusters);
    }
    // return an array of indices, representing the nearest "centroid" to each data point
    return nearest;
  });
};

function initCentroids(X: number[][], nClusters: number): tf.Tensor {
  const centroids = [];
  for (let i = 0; i < nClusters; i++) {
    centroids.push([]);
  }
  for (let j = 0; j < X[0].length; j++) {
    const col = X.map(instance => instance[j]);
    const max = Math.max(...col);
    const min = Math.min(...col);
    for (let i = 0; i < nClusters; i++) {
      centroids[i].push(Math.random() * (max - min) + min)
    }
  }
  return tf.tensor2d(centroids);
}

export function assignToNearest(samples: tf.Tensor2D, centroids: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    // perform outer operations by expanding dimensions
    const expandedVectors = tf.expandDims(samples, 0);
    const expandedCentroids = tf.expandDims(centroids, 1);
    const distances = tf.sum(
      tf.square(
        tf.sub(expandedVectors, expandedCentroids)
      ), 2
    );
    const mins = tf.argMin(distances, 0);
    return mins.toInt();
  });
}

function updateCentroids(samples: tf.Tensor2D, nearestIndices: tf.Tensor, nClusters: number): tf.Tensor {
  return tf.tidy(() => {
    const newCentroids = [];
    for (let i = 0; i < nClusters; i++) {
      const mask = tf.equal(nearestIndices, tf.scalar(i).toInt());
      const currentCentroid = tf.div(
        // set all masked instances to 0 by multiplying the mask tensor, then sum across all instances
        tf.sum(tf.mul(tf.expandDims(mask.toFloat(), 1), samples), 0),
        // divided by number of instances
        tf.sum(mask.toFloat())
      );
      newCentroids.push(currentCentroid);
    }
    return tf.stack(newCentroids);
  });
}

export function genRandomSamples(nClusters: number, nSamplesPerCluster: number,
                                 nFeatures = 2, embiggenFactor = 5, seed = 0) {
  return tf.tidy(() => {
    let slices = [];
    let centroidsArr: tf.Tensor[] = [];
    // Create samples for each cluster
    for (let i = 0; i < nClusters; i++) {
      let samples = tf.randomNormal([nSamplesPerCluster, nFeatures], 0, 5, 'float32', seed);

      const currentCentroid = tf.randomUniform([1, nFeatures], embiggenFactor * -8, embiggenFactor * 8);
      samples = samples.add(currentCentroid);
      centroidsArr = centroidsArr.concat(currentCentroid);
      slices.push(samples)
    }
    // Create a big "samples" dataset
    const samples = tf.concat(slices, 0);
    const centroids = tf.concat(centroidsArr, 0);
    return {
      centroids,
      samples
    }
  })
}
