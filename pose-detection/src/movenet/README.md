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
3.  [Performance](#performance)

## Installation

Via script tags:

```html
<!-- Require the peer dependencies of pose-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
<!-- Alternatively you can use the WASM backend: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"></script> -->

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/pose-detection
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
```

Install one of the backends:
WebGL:
```sh
yarn add @tensorflow/tfjs-backend-webgl
```

WASM:
```sh
yarn add @tensorflow/tfjs-backend-wasm
```

-------------------------------------------------------------------------------

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

`detectorConfig` is an object that defines MoveNet specific configurations:

*   *modelType* (optional): specify which MoveNet variant to load from the
    `poseDetection.movenet.modelType` enum list. Default to
    `poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING`.

*   *enableSmoothing*: A boolean indicating whether to use temporal filter to
    smooth the predicted keypoints. Defaults to *True*. The temporal filter
    relies on the `currentTime` field of the `HTMLVideoElement`. You can
    override this timestamp by passing in your own timestamp (in microseconds)
    as the third parameter. This is useful when the input is a tensor, which
    doesn't have the `currentTime` field. Or in testing, to simulate different FPS.

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
`HTMLCanvasElement`.

The following code snippet demonstrates how to run the model inference:

```javascript
const poses = await detector.estimatePoses(image);
```

Please refer to the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation)
about the structure of the returned `poses`.

## Performance
To quantify the inference speed of MoveNet, the model was benchmarked across
multiple devices. The model latency (expressed in FPS) was measured on GPU with
WebGL, as well as WebAssembly (WASM), which is the typical backend for devices
with lower-end or no GPUs.

|              | MacBook Pro 15" 2019 <br> Intel core i9. <br> AMD Radeon Pro Vega 20 Graphics. <br> (FPS) | iPhone 12 <br> (FPS) | Pixel 5 <br> (FPS) | Desktop <br> Intel i9-10900K. <br> Nvidia GTX 1070 GPU. <br> (FPS) |
| --- | --- | --- | --- | --- |
|       *WebGL*                        |  104 \| 77 | 51 \| 43 | 34 \| 12 | 87 \| 82 |
|  *WASM* <br> with SIMD + Multithread |  42 \| 21 | N/A | N/A | 71 \| 30 |

To see the model’s FPS on your device, try our
[demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet).
You can switch the model type and backends live in the demo UI to see what works
best for your device.
