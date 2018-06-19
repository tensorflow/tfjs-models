# KNN Image Classifier Demo

## Contents

### Demo: Camera

The camera demo shows how to create a custom classifier with 3 classes that can be trained in realtime using a webcamera. Hold down the train button to add samples to the classifier, and then let it predict which of the 3 classes that is closest.

## Setup

cd into the demos folder:

```sh
cd knn-classifier/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing the classifier locally, and want to test the changes in the demos

Install yalc:
```sh
npm i -g yalc
```

cd into the knn-classifier folder:
```sh
cd knn-classifier
```

Install dependencies:
```sh
yarn
```

Publish knn-classifier locally:
```sh
yalc push
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Start the dev demo server:
```sh
yarn watch
```
