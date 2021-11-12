# SelfieSegmentation

SelfieSegmentation-TFJS uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.

Two models are offered.

* general - implementation operates on a 256x256x3 tensor, and outputs a 256x256x1 tensor representing the segmentation mask
* landscape - similar to the general model, but operates on a 144x256x3 tensor. It has fewer FLOPs than the general model, and therefore, runs faster.

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/body-segmentation/index.html?model=selfie_segmentation).
In the runtime-backend dropdown, choose 'tfjs-webgl'.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use SelfieSegmentation, you need to first select a runtime (TensorFlow.js or MediaPipe).
This guide is for TensorFlow.js
runtime. The guide for MediaPipe runtime can be found
[here](https://github.com/tensorflow/tfjs-models/tree/master/body-segmentation/src/mediapipe).

Via script tags:

```html
<!-- Require the peer dependencies of body-segmentation. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/body-segmentation"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/body-segmentation
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
```

-----------------------------------------------------------------------
## Usage

If you are using the Body Segmentation API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tf from '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
```
### Create a detector

Pass in `bodySegmentation.SupportedModels.SelfieSegmentation` from the
`bodySegmentation.SupportedModel` enum list along with a `segmenterConfig` to the
`createDetector` method to load and initialize the model.

`segmenterConfig` is an object that defines MediaPipeHands specific configurations for `SelfieSegmentationMediaPipeModelConfig`:

*   *runtime*: Must set to be 'tfjs'.

*   *modelType*: specify which variant to load from `SelfieSegmentationModelType` (i.e.,
    'general', 'landscape'). If unset, the default is 'general'.

*   *modelUrl*: An optional string that specifies custom url of
the segmentation model. This is useful for area/countries that don't have access to the model hosted on tf.hub.

```javascript
const model = bodySegmentation.SupportedModels.SelfieSegmentation;
const segmenterConfig = {
  runtime: 'tfjs',
};
segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);
```

### Run inference

Now you can use the segmenter to segment hand poses. The `segmentPeople` method
accepts both image and video in many formats, including: `tf.Tensor3D`,
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `segmentationConfig` parameter.

`segmentationConfig` is an object that defines SelfieSegmentation specific configurations for `SelfieSegmentationTfjsSegmentationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

The following code snippet demonstrates how to run the model inference:

```javascript
const segmentationConfig = {flipHorizontal: false};
const people = await segmenter.segmentPeople(image, segmentationConfig);
```

The returned `people` array contains a single element only, where all the people segmented in the image are found in that single segmentation element.

The only labels returned by the maskValueToLabel function by the model is 'upper_body'.

Please refer to the Body Segmentation API
[README](https://github.com/tensorflow/tfjs-models/blob/master/body-segmentation/README.md#how-to-run-it)
about the structure of the returned `people`.
