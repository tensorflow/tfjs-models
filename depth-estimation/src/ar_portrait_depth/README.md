# AR Portrait Depth

This portrait depth model estimates per-pixel depth (the distance to the camera center) for a single portrait image, which can be further used for creative applications. (See [DepthLab](https://augmentedperception.github.io/depthlab/) for potential applications). Note that the model runs locally on the user’s device and no data is uploaded to the server.

For example, the following demo transforms a single 2D RGB image into a 3D Portrait: [3D Photo Demo](https://storage.googleapis.com/tfjs-models/demos/3dphoto).

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use ARPortraitDepth:

Via script tags:

```html
<!-- Require the peer dependencies of depth-estimation. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<!-- WebGL is the recommended backend. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/body-segmentation"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/depth-estimation"></script>
```

Via npm:
```sh
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
yarn add @tensorflow-models/body-segmentation
yarn add @tensorflow-models/depth-estimation
```

-----------------------------------------------------------------------
## Usage

If you are using the depth-estimation API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-converter';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow-models/body-segmentation';
import * as depthEstimation from '@tensorflow-models/depth-estimation';
```

### Create an estimator

Pass in `depthEstimation.SupportedModels.ARPortraitDepth` from the
`depthEstimation.SupportedModel` enum list along with an `estimatorConfig` to the
`createEstimator` method to load and initialize the model.

`estimatorConfig` is an object that defines ARPortraitDepth specific configurations for `ARPortraitDepthModelConfig`:

*   *segmentationModelUrl*: An optional string that specifies custom url of
the segmenter model. This is useful for area/countries that don't have access to the model hosted on tf.hub. It also accepts `io.IOHandler` which can be used with
[tfjs-react-native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
to load model from app bundle directory using
[bundleResourceIO](https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/

*   *depthModelUrl*: An optional string that specifies custom url of
the estimator model. This is useful for area/countries that don't have access to the model hosted on tf.hub. It also accepts `io.IOHandler` which can be used with
[tfjs-react-native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
to load model from app bundle directory using
[bundleResourceIO](https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/

```javascript
const model = depthEstimation.SupportedModels.ARPortraitDepth;
const estimatorConfig = {
  outputDepthRange: [0, 1]
};
estimator = await depthEstimation.createEstimator(model, estimatorConfig);
```

### Run inference

Now you can use the estimator to estimate the depth. The `estimateDepth` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is an object that defines ARPortraitDepth specific configurations for `ARPortraitDepthEstimationConfig`:

*   *minDepth*: The minimum depth value for the model to map to 0. Any smaller
depth values will also get mapped to 0.

*   *maxDepth*: The maximum depth value for the model to map to 1. Any larger
depth values will also get mapped to 1.

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {flipHorizontal: false};
const depthMap = await estimator.estimateDepth(image, estimationConfig);
```

Please refer to the Depth Estimation API
[README](https://github.com/tensorflow/tfjs-models/blob/master/depth-estimation/README.md#how-to-run-it)
about the structure of the returned `depthMap`.
