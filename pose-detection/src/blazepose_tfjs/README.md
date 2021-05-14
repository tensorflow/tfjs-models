# BlazePose

BlazePoseTfjs uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.
Three models are offered.

* 'lite' - our smallest model which trades footprint for accuracy.
* 'heavy' - our largest model intended for high accuracy, regardless of size.
* 'full' - A middle ground between Lite and Heavy.

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose).
In the runtime-backend dropdown, choose 'tfjs-webgl'.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)
3.  [Example Code and Demos](#example-code-and-demos)

## Installation

Please follow the instructions in the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#installation)
to install the package.

## Usage

If you are using the Pose API via npm, you need to import the libraries first.

### Import the library

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-wasm';
```


### Create a detector

```javascript
const model = poseDetection.SupportedModels.BlazePose;
const detectorConfig = {
  runtime: 'tfjs'
};
detector = await poseDetection.createDetector(model, detectorConfig);
```
export interface BlazePoseModelConfig extends ModelConfig {
  // Defaults to 'mediapipe' if not provided.
  runtime?: 'mediapipe'|'tfjs';
  enableSmoothing?: boolean;
  modelType?: BlazePoseModelType;
}
Pass in `poseDetection.SupportedModels.BlazePose` from the
`posedetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is a dictionary that defines BlazePose specific configurations.
For BlazePoseTfjs:

*   *backend*: Must be set to 'tfjs' to use this model.
*   *enableSmoothing*: Defaults to true, but you can turn this off by setting
    this option to false.
*   *modelType*: specify which variant to load from BlazePoseModelType (i.e.,
    'lite', 'full', 'heavy'). If unset, the default is 'full'.

### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method
accepts both image and video in many formats, including: `tf.Tensor3D`,
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
