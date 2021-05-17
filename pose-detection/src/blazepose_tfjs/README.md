# BlazePose

BlazePose TFJS uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.

Three models are offered.

* lite - our smallest model that is less accurate but smaller in model size and minimal memory footprint.
* heavy - our largest model intended for high accuracy, regardless of size.
* full - A middle ground between performance and accuracy.

Please try it out using the live [demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose).
In the runtime-backend dropdown, choose 'tfjs-webgl'.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)
3.  [Example Code and Demos](#example-code-and-demos)

## Installation

Via script tags:

```html
<!-- Require the peer dependencies of pose-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/pose-detection
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
```

-----------------------------------------------------------------------
## Usage

If you are using the Pose API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
```
### Create a detector

Pass in `poseDetection.SupportedModels.BlazePose` from the
`posedetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines BlazePose specific configurations for `BlazePoseTfjsModelConfig`:

*   *backend*: Must be set to 'tfjs' to use this model.
*   *enableSmoothing*: Defaults to true, but you can turn this off by setting
    this option to false.
*   *modelType*: specify which variant to load from BlazePoseModelType (i.e.,
    'lite', 'full', 'heavy'). If unset, the default is 'full'.

```javascript
const model = poseDetection.SupportedModels.BlazePose;
const detectorConfig = {
  runtime: 'tfjs'
};
detector = await poseDetection.createDetector(model, detectorConfig);
```
export interface BlazePoseModelConfig extends ModelConfig {
  runtime: 'mediapipe'|'tfjs';
  enableSmoothing?: boolean;
  modelType?: BlazePoseModelType;
}
### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method
accepts either image or video in many formats, including: `tf.Tensor3D`,
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `estimationConfig` parameter.

The following code snippet demonstrates how to run the model inference:

```javascript
const poses = await detector.estimatePoses(image);
```

Please refer to the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation)
about the structure of the returned `poses`.

## Example Code and Demos

You may reference the demos for code examples. Details for how to run the demos
are included in the demo
[folder](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/demo).
