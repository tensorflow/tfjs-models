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

* *modelType* (optional): specify which MoveNet variant to load from the
  `poseDetection.movenet.modelType` enum list:
  * `SINGLEPOSE_LIGHTNING`. Default. The fastest single-pose detector.
  * `SINGLEPOSE_THUNDER`. A more accurate but slower single-pose detector.
  * `MULTIPOSE_LIGHTNING`. Multi-pose detector that detects up to 6 poses.

* *enableSmoothing* (optional): A boolean indicating whether to use temporal
  filter to smooth the predicted keypoints. Defaults to *True*. The temporal
  filter relies on the `currentTime` field of the `HTMLVideoElement`. You can
  override this timestamp by passing in your own timestamp (in milliseconds)
  as the third parameter. This is useful when the input is a tensor, which
  doesn't have the `currentTime` field. Or in testing, to simulate different FPS.

* *modelUrl* (optional): An optional string that specifies custom url of the
  MoveNet model. If not provided, it will load the model specified by
  *modelType* from tf.hub. This argument is useful for area/countries that
  don't have access to the model hosted on tf.hub. It also accepts
  `io.IOHandler` which can be used with
  [tfjs-react-native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
  to load model from app bundle directory using
  [bundleResourceIO](https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/src/bundle_resource_io.ts#L169).

* *minPoseScore* (optional): The minimum confidence score a pose needs to have
  to be considered a valid pose detection.

* *multiPoseMaxDimension* (optional): The target maximum dimension to use as the
  input to the multi-pose model. Must be a multiple of 32 and defaults to 256.
  The recommended range is [128, 512]. A higher maximum dimension results in
  higher accuracy but slower speed, whereas a lower maximum dimension results in
  lower accuracy but higher speed. The input image will be resized so that its
  maximum dimension will be the given number, while maintaining the input image
  aspect ratio. As an example: with 320 as the maximum dimension and a 640x480
  input image, the model will resize the input to 320x240. A 720x1280 image will
  be resized to 180x320.

* *enableTracking* (optional): A boolean indicating whether detected persons
  will be tracked across frames. If true, each pose will have an ID that
  uniquely identifies a person. Only used with multi-pose models.

  For more information about tracking, see the documentation
  [here](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/calculators/tracker.md).

* *trackerType* (optional): A `TrackerType` indicating which type of tracker to
  use. Defaults to bounding box tracking.

* *trackerConfig* (optional): A `TrackerConfig` object that specifies the
  configuration to use for the tracker. For properties that are not specified,
  default values will be used.

The following code snippet demonstrates how to load the
**MoveNet.SinglePose.Lightning** model:

```javascript
const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
```

The following code snippet demonstrates how to load the
**MoveNet.MultiPose.Lightning** model with bounding box
[tracking](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/calculators/tracker.md)
enabled:

```javascript
const detectorConfig = {
  modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
  enableTracking: true,
  trackerType: poseDetection.TrackerType.BoundingBox
};
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
for the basic structure of the returned `poses`. When running the multi-pose
MoveNet model the `box` field in a returned `Pose` will be set with a bounding
box around the detected person. When tracking is enabled, the `id` field of a
`Pose` will contain a unique ID that identifies a tracked person.

## Performance
To quantify the inference speed of MoveNet, the model was benchmarked across
multiple devices. The model latency (expressed in FPS) was measured on GPU with
WebGL, as well as WebAssembly (WASM), which is the typical backend for devices
with lower-end or no GPUs.

SinglePose Lightning | SinglePose Thunder | Multipose Lightning

|              | MacBook Pro 15" 2019 <br> Intel core i9. <br> AMD Radeon Pro Vega 20 Graphics. <br> (FPS) | iPhone 12 <br> (FPS) | Pixel 5 <br> (FPS) | Desktop <br> Intel i9-10900K. <br> Nvidia GTX 1070 GPU. <br> (FPS) |
| --- | --- | --- | --- | --- |
|       *WebGL*                        |  104 \| 77 \| 54 | 51 \| 43 \| 24 | 34 \| 12 \| 8 | 87 \| 82 \| 62 |
|  *WASM* <br> with SIMD + Multithread |  42 \| 21 \| N/A | N/A | N/A | 71 \| 30 \| N/A |

Note that for multi-person detection, the number of detected persons does not
impact inference speed and the accuracy of detections is similar to that of
SinglePose Lightning.

To see the model’s FPS on your device, try our
[demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet).
You can switch the model type and backends live in the demo UI to see what works
best for your device.
