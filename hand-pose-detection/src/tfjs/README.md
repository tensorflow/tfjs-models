# MediaPipeHands

MediaPipeHands-TFJS uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.

Two models are offered.

* lite - our smallest model that is less accurate but smaller in model size and minimal memory footprint.
* full - A middle ground between performance and accuracy.

Please try it out using the live [demo](https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection/index.html?model=mediapipe_hands).
In the runtime-backend dropdown, choose 'tfjs-webgl'.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use MediaPipeHands, you need to first select a runtime (TensorFlow.js or MediaPipe).
This guide is for TensorFlow.js
runtime. The guide for MediaPipe runtime can be found
[here](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection/src/mediapipe).

Via script tags:

```html
<!-- Require the peer dependencies of hand-pose-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/hand-pose-detection
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
```

-----------------------------------------------------------------------
## Usage

If you are using the Hands API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
```
### Create a detector

Pass in `handPoseDetection.SupportedModels.MediaPipeHands` from the
`handPoseDetection.SupportedModel` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines MediaPipeHands specific configurations for `MediaPipeHandsTfjseModelConfig`:

*   *runtime*: Must set to be 'tfjs'.

*   *maxHands*: Defaults to 2. The maximum number of hands that will be detected by the model. The number of returned hands can be less than the maximum (for example when no hands are present in the input). It is highly recommended to set this value to the expected max number of hands, otherwise the model will continue to search for the missing hands which can slow down the performance.

*   *modelType*: specify which variant to load from `MediaPipeHandsModelType` (i.e.,
    'lite', 'full'). If unset, the default is 'full'.

*   *detectorModelUrl*: An optional string that specifies custom url of
the detector model. This is useful for area/countries that don't have access to the model hosted on tf.hub.
*   *landmarkModelUrl* An optional string that specifies custom url of
the landmark model. This is useful for area/countries that don't have access to the model hosted on tf.hub.

```javascript
const model = handPoseDetection.SupportedModels.MediaPipeHands;
const detectorConfig = {
  runtime: 'tfjs',
};
detector = await handPoseDetection.createDetector(model, detectorConfig);
```

### Run inference

Now you can use the detector to detect hand poses. The `estimateHands` method
accepts both image and video in many formats, including: `tf.Tensor3D`,
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is an object that defines MediaPipeHands specific configurations for `MediaPipeHandsTfjsEstimationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

*   *staticImageMode*: Optional. Defaults to false. If set to true, hand pose detection
will run on every input image, otherwise if set to false then detection runs
once and then the model simply tracks those landmarks without invoking
another detection until it loses track of any of the hands (ideal for videos).

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {flipHorizontal: false};
const hands = await detector.estimateHands(image, estimationConfig);
```

Please refer to the Hands API
[README](https://github.com/tensorflow/tfjs-models/blob/master/hand-pose-detection/README.md#how-to-run-it)
about the structure of the returned `hands`.
