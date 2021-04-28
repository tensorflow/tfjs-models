# MoveNet

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
The model is offered on TF.Hub with two variants,
[Lightning](https://tfhub.dev/google/movenet/singlepose/lightning/3) and
[Thunder](https://tfhub.dev/google/movenet/singlepose/thunder/3). Lightning is
intended for latency-critical applications, and Thunder is intended for
applications that require high accuracy. Both models run faster than real time
(30+ FPS) on most modern desktops and laptops, which proves crucial for live
fitness, health, and wellness applications. Please try it out using the live
[demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet).

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

### Import the libraries

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-wasm';
```

### Create a detector

Pass in `poseDetection.SupportedModels.MoveNet` from the
`posedetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the MoveNet model.

`detectorConfig` is a dictionary that defines MoveNet specific configurations:

*   *modelType*: specify which MoveNet variant to load from the
    `poseDetection.movenet.modelType` enum list.

*   *modelUrl* (optional): An optional string that specifies custom url of the
	MoveNet model. If not provided, it will load the model specified by 
	*modelType* from tf.hub. This argument is useful for area/countries that 
	don't have access to the model hosted on tf.hub.

The following code snippet demonstrates how to load the
**MoveNet.SinglePose.Lightning** model:

```javascript
const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
```

### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method 
accepts both image and video in many formats, including:
`tf.Tensor3D`, `ImageData`, `HTMLVideoElement`, `HTMLImageElement`,
`HTMLCanvasElement`. If you want more options, you can pass in an
`estimationConfig` as the second parameter.

`estimationConfig` is a dictionary that defines the parameters used by MoveNet
at inference time:

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