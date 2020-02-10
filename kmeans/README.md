# KMeans Clustering

This package provides a utility for doing unsupervised learning using the
[K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
algorithm.

This package is different from the other packages in this repository in that it
doesn't provide a model with weights, but rather an algorithm for grouping data
points together based on similarities in their feature space, i.e. unsupervised
machine learning.

You can see example code [here](https://github.com/tensorflow/tfjs-models/tree/master/kmeans/demo).

## Usage example

##### via Script Tag

```html
<html>
  <head>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.5.0"></script>
    <!-- Load KMeans -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/kmeans@0.1.0"></script>
  </head>

  <body></body>
  <!-- Place your code in the script tag below. You can also use an external .js file -->
  <script>
    const init = async function() {
      // Create the data
      const nSamplesPerCluster = 100;
      const nFeatures = 3;
      const nClusters = 4;
      const dataCollection = [];

      // Generate samples for each cluster
      for (let i = 0; i < nClusters; i++) {
        const currentCentroid = tf.randomUniform([1, nFeatures]);
        const data = tf
          .randomNormal([nSamplesPerCluster, nFeatures])
          .add(currentCentroid);
        dataCollection.push(data);
      }
      // Combine into one dataset
      const X = tf.concat(dataCollection, 0);

      // Create the model
      const model = kmeans.kMeans({nClusters});

      // Train and predict on training data
      const y = model.fitPredict(X);

      const yData = await y.data();
      console.log(yData);
    };

    init();
  </script>
</html>
```

###### via NPM

```js
import * as tf from '@tensorflow/tfjs';
import {kMeans} from '@tensorflow-models/kmeans';

const init = async function() {
  // Create the data
  const nSamplesPerCluster = 100;
  const nFeatures = 3;
  const nClusters = 4;
  const dataCollection = [];

  // Generate samples for each cluster
  for (let i = 0; i < nClusters; i++) {
    const currentCentroid = tf.randomUniform([1, nFeatures]);
    const data = tf
      .randomNormal([nSamplesPerCluster, nFeatures])
      .add(currentCentroid);
    dataCollection.push(data);
  }
  // Combine into one dataset
  const X = tf.concat(dataCollection, 0);

  // Create the model
  const model = kMeans({nClusters});

  // Train and predict on training data
  const y = model.fitPredict(X);

  const yData = await y.data();
  console.log(yData);
```

## API

#### Creating a classifier

`kmeans` is the module name, which is automatically included when you use
the <script src> method. `kMeans` is the model constructor name, which can be
used to create and instance of `KMeansClustering`. To do so, pass a `KMeansArgs`
to the condtructor.

```ts
import {kMeans} from '@tensorflow-models/kmeans';
const model = kMeans({nClusters: 8});
console.log(model instanceof KMeansClustering, KMeansClustering); // true
```

Returns a `KMeans` clustering instance.

#### Changing hyper-parameters

```ts
const model = kMeans({
  nClusters: number,
  maxIter: number,
  tol: number,
}): KMeansClustering;
```

Args:

- **nClusters:** Number of clusters to group input datasets into. Optional,
default to 8.
- **maxIter:** Maximum number of iterations for updating the centroids before
the algorithm concludes. Optional, default to 300.
- **tol:** Tolerance for the difference of centroids' positions between two
model iterations. Model stops when the difference becomes lower than this value.
Useful for early stopping. Optional, default to 10e-4.

#### Training

```ts
model.fit(x: tf.Tensor): tf.Tensor;
// or:
model.fitPredict(x: tf.Tensor): tf.Tensor;
```

Args:

- **x:** Training data, a two-dimensional tensor with shape
`[nInstances, nFeatures]`.

Returns a vector representing the prediction on training data `x`, with shape
`[nInstances, 1]`. Each dimension of the vector is an integer, representing the
cluster ID a data instance belongs to.

#### Incremental training

```ts
model.fitOneCycle(x: tf.Tensor): tf.Tensor;
```

Args:

- **x:** Training data, a two-dimensional tensor with shape
`[nInstances, nFeatures]`.

Returns a vector representing the prediction on training data `x`, with shape
`[nInstances, 1]`. Each dimension of the vector is an integer, representing the
cluster ID a data instance belongs to.

Different from `model.fit` or `model.fitPredict`, this method only runs one
iteration each function call. Useful for demoing or debugging.

#### Making a prediction

```ts
model.predict(x: tf.Tensor): tf.Tensor;
```

Args:

- **x:** Test data, a tensor in shape `[nInstances, nFeatures]`.

Returns a vector representing the prediction on test data `x`, with shape
`[nInstances, 1]`. Each dimension of the vector is an integer, representing the
cluster ID a data instance belongs to.

