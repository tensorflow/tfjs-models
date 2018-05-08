# Pre-trained TensorFlow.js models

This repository hosts a set of pre-trained models that have been ported to
TensorFlow.js and have been hosted on NPM and unpkg so they can be used in any
project out of the box. They can be used directly or used in a transfer learning
setting with TensorFlow.js.

To find out about APIs for models, look at the README in each of the respective
directories. In general, we try to hide tensors so the API can be used by
non-machine learning experts.

If you want to contribute a model, please first file an issue to find out
whether we will accept it. We are trying to only add models that can be used
as building blocks in other applications.

## Models

### Image classification
- [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)
  - `npm install @tensorflow-models/mobilenet`
- [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet)
  - `npm install @tensorflow-models/posenet`
