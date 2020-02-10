# KMeans Clustering Demo

## Contents

### Demo: Clustering two-dimensional data

The two-dimensional data clustering demo shows how to set up a model that groups
data into `k = 4` clsters, and how the cluster centroids (parameters of the
KMeans model) are learned during each iteration.

To play with the demo:

- Click "Regen Training Data" to generate new training data. Colors of points
  represent diffenrent two-dimensional normal distributions according to which the
  data points are generated -- the "true" clusters. Cluster centers are shown with
  black "+" marks.

- Then click "Fit" to train the model on the generated training data; or click
  "Fit One-Cycle" to see a step-by-step view of how the model is trained. Now,
  colors of points represent the predicted cluster of each point. Predicted
  centroids are represented as coloful "+" marks.

- Next click on ”New Test Data" to generate new data and make predictions using
  the trained model. Color represent the same clusters as for the training data.

## Setup

cd into the demos folder:

```sh
cd kmeans/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing the algorithm locally, and want to test the changes in the demos

cd into the kmeans/ folder:

```sh
cd kmeans
```

Install dependencies:

```sh
yarn
```

Publish kmeans locally:

```sh
yarn publish-local
```

cd into this directory, kmeans/demos and install dependencies:

```sh
cd demos
yarn
```

Link the package published from the publish step above:

```sh
yarn link-local
```

Start the dev demo server:

```sh
yarn watch
```
