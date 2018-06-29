# Pre-trained TensorFlow.js models

This repository hosts a set of pre-trained models that have been ported to
TensorFlow.js.

The models are hosted on NPM and unpkg so they can be used in any project out of the box. They can be used directly or used in a transfer learning
setting with TensorFlow.js.

To find out about APIs for models, look at the README in each of the respective
directories. In general, we try to hide tensors so the API can be used by
non-machine learning experts.

For those intested in contributing a model, please file a [GitHub issue on tfjs](https://github.com/tensorflow/tfjs/issues) to gauge
interest. We are trying to add models that complement the existing set of models
and can be used as building blocks in other apps.

## Models

### Image classification
- [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet) - Classify images with labels from the [ImageNet database](http://www.image-net.org/).
  - `npm i @tensorflow-models/mobilenet`
- [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) - Realtime pose detection. Blog post [here](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5).
  - `npm i @tensorflow-models/posenet`

### General utilities
- [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier) - Create a custom k-nearest neighbors classifier. Can be used for transfer learning.
  - `npm i @tensorflow-models/knn-classifier`

## Development

You can run the unit tests for any of the models by running the following
inside a directory:

`yarn test`

New models should have a test NPM script.

To run all of the tests, you can run the following command from the root of this
repo:

`yarn presubmit`
