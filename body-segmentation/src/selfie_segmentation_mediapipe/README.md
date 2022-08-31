# MediaPipe SelfieSegmentation

MediaPipe SelfieSegmentation-MediaPipe wraps the MediaPipe JS Solution within the familiar
TFJS API [mediapipe.dev](https://mediapipe.dev).

Two variants of the model are offered.

* general - implementation operates on a 256x256x3 tensor, and outputs a 256x256x1 tensor representing the segmentation mask
* landscape - similar to the general model, but operates on a 144x256x3 tensor. It has fewer FLOPs than the general model, and therefore, runs faster.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use MediaPipe SelfieSegmentation:

Via script tags:

```html
<!-- Require the peer dependencies. -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/body-segmentation"></script>
```

Via npm:
```sh
yarn add @mediapipe/selfie_segmentation
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-backend-webgl
yarn add @tensorflow-models/body-segmentation
```

-----------------------------------------------------------------------
## Usage

If you are using the Body Segmentation API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import '@mediapipe/selfie_segmentation';
```

### Create a detector

Pass in `bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation` from the
`bodySegmentation.SupportedModel` enum list along with a `segmenterConfig` to the
`createSegmenter` method to load and initialize the model.

`segmenterConfig` is an object that defines MediaPipeSelfieSegmentation specific configurations for `MediaPipeSelfieSegmentationMediaPipeModelConfig`:

*   *runtime*: Must set to be 'mediapipe'.

*   *modelType*: specify which variant to load from `MediaPipeSelfieSegmentationModelType` (i.e.,
    'general', 'landscape'). If unset, the default is 'general'.

*   *solutionPath*: The path to where the wasm binary and model files are located.

```javascript
const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
const segmenterConfig = {
  runtime: 'mediapipe',
  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation'
                // or 'base/node_modules/@mediapipe/selfie_segmentation' in npm.
};
segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);
```

### Run inference

Now you can use the segmenter to segment people. The `segmentPeople` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`, `ImageData`, `Tensor3D`. If you want more
options, you can pass in a second `segmentationConfig` parameter.

`segmentationConfig` is an object that defines MediaPipeSelfieSegmentation specific configurations for `MediaPipeSelfieSegmentationMediaPipeSegmentationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

The following code snippet demonstrates how to run the model inference:

```javascript
const segmentationConfig = {flipHorizontal: false};
const people = await segmenter.segmentPeople(image, segmentationConfig);
```

The returned `people` array contains a single element only, where all the people segmented in the image are found in that single segmentation element.

The only label returned by the maskValueToLabel function is 'person'.

Please refer to the Body Segmentation API
[README](https://github.com/tensorflow/tfjs-models/blob/master/body-segmentation/README.md#how-to-run-it)
about the structure of the returned `people` array.
