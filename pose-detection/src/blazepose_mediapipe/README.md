# BlazePose

BlazePose-MediaPipe wraps our powerful MediaPipe JS Solution within the familiar
TFJS API [mediapipe.dev](https://mediapipe.dev). Three models are offered.

* 'lite' - our smallest model which trades footprint for accuracy.
* 'heavy' - our largest model intended for high accuracy, regardless of size.
* 'full' - A middle ground between Lite and Heavy.

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose).

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

For NPM, you will install the module. At the time of this release, the supported
version is 0.3.x.

```bash
npm install @mediapipe/pose@0.3.x
```

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
```


### Create a detector

```javascript
const model = poseDetection.SupportedModels.BlazePose;
const detectorConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/pose'
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
`createDetector` method to load and initialize the MediaPipe model.

`detectorConfig` is a dictionary that defines BlazePose specific configurations.
For BlazePose-MediaPipe:

*   *backend*: Must be set to 'mediapipe' to use this model.
*   *solutionPath*: The path to where the additional model files are located on
    your server.
*   *enableSmoothing*: Defaults to true, but you can turn this off by setting
    this option to false.
*   *modelType*: specify which variant to load from BlazePoseModelType (i.e.,
    'lite', 'full', 'heavy'). If unset, the default is 'full'.

### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is a dictionary that defines the parameters used by
BlazePose-MediaPipe at inference time:

*   *enableSmoothing*: A boolean indicating whether to use temporal filter to
    smooth the predicted keypoints. Defaults to *True*. The temporal filter
    relies on the `currentTime` field of the `HTMLVideoElement`. You can
    override this timestamp by passing in your own timestamp (in microseconds)
    as the third parameter. This is useful when the input is a tensor, which
    doesn't have the `currentTime` field. Or in testing, to simulate different FPS.

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {enableSmoothing: true};
const poses = await detector.estimatePoses(image, estimationConfig);
```

Please refer to the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation)
about the structure of the returned `poses`.

## Example Code and Demos

You may reference the demos for code examples. Details for how to run the demos
are included in the demo
[folder](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/demo).
