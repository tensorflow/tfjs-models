# BlazePose

BlazePose-MediaPipe wraps the powerful MediaPipe JS Solution within the familiar
TFJS API [mediapipe.dev](https://mediapipe.dev). Three models are offered.

* lite - our smallest model which trades footprint for accuracy.
* heavy - our largest model intended for high accuracy, regardless of size.
* full - A middle ground between Lite and Heavy.

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose).

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)
3.  [Performance](#performance)
4.  [Bundle Size](#bundle-size)

## Installation

To use BlazePose, you need to first select a runtime (TensorFlow.js or MediaPipe).
To understand the advantages of each runtime, check the performance
and bundle size section for further details. This guide is for MediaPipe
runtime. The guide for TensorFlow.js runtime can be found
[here](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/src/blazepose_tfjs).

Via script tags:

```html
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/pose-detection
yarn add @mediapipe/pose
```

-----------------------------------------------------------------------
## Usage

If you are using the Pose API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@mediapipe/pose';
```

### Create a detector
Pass in `poseDetection.SupportedModels.BlazePose` from the
`posedetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines BlazePose specific configurations for `BlazePoseMediaPipeModelConfig`:

*   *runtime*: Must set to be 'mediapipe'.

*   *enableSmoothing*: Defaults to true. If your input is a static image, set it to false. This flag is used to indicate whether to use temporal filter to smooth the predicted keypoints.

*   *modelType*: specify which variant to load from `BlazePoseModelType` (i.e.,
    'lite', 'full', 'heavy'). If unset, the default is 'full'.

*   *solutionPath*: The path to where the wasm binary and model files are located.

```javascript
const model = poseDetection.SupportedModels.BlazePose;
const detectorConfig = {
  runtime: 'mediapipe',
  solutionPath: 'base/node_modules/@mediapipe/pose'
};
detector = await poseDetection.createDetector(model, detectorConfig);
```

### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is an object that defines BlazePose specific configurations for `BlazePoseMediaPipeEstimationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

You can also override a video's timestamp by passing in a timestamp in
milliseconds as the third parameter.

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {enableSmoothing: true};
const poses = await detector.estimatePoses(image, estimationConfig);
```

Please refer to the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation)
about the structure of the returned `poses`.

## Performance
To quantify the inference speed of MoveNet, the model was benchmarked across
multiple devices. The model latency (expressed in FPS) was measured on GPU with
WebGL, as well as WebAssembly (WASM), which is the typical backend for devices
with lower-end or no GPUs.

|  |MacBook Pro 15" 2019<br>Intel core i9.<br>AMD Radeon Pro Vega 20 Graphics.<br> (FPS)| iPhone12<br>(FPS) | Pixel5 <br> (FPS)|Desktop <br> Intel i9-10900K. <br> Nvidia GTX 1070 GPU. <br> (FPS)|
| --- | --- | --- | --- | --- |
|       *MediaPipe Runtime* <br> With WASM & GPU Accel.                        |  92 \| 81 \| 38 | N/A | 32 \| 22 \| N/A | 160 \| 140 \| 98 |
|  *TensorFlow.js Runtime* <br> with WebGL backend |  48 \| 53 \| 28 | 34 \| 30 \| N/A | 12  \| 11 \| 5 | 44 \| 40 \| 30 |

To see the model’s FPS on your device, try our
[demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose).
You can switch the model type and backends live in the demo UI to see what works
best for your device.

## Bundle Size
Bundle size can affect initial page loading experience, such as Time-To-Interactive (TTI), UI rendering, etc. We evaluate the pose-detection API and the two runtime options. The bundle size affects file fetching time and UI smoothness, because processing the code and loading them into memory will compete with UI rendering on CPU. It also affects when the model is available to make inference.

There is a difference of how things are loaded between the two runtimes. For the MediaPipe runtime, only the @tensorflow-models/pose-detection and the @mediapipe/pose library are loaded at initial page download; the runtime and the model assets are loaded when the createDetector method is called. For the TF.js runtime with WebGL backend, the runtime is loaded at initial page download; only the model assets are loaded when the createDetector method is called. The TensorFlow.js package sizes can be further reduced with a custom bundle technique. Also, if your application is currently using TensorFlow.js, you don’t need to load those packages again, models will share the same TensorFlow.js runtime. Choose the runtime that best suits your latency and bundle size requirements. A summary of loading times and bundle sizes is provided below:


|  |Bundle Size<br>gzipped + minified|Average Loading Time <br> download speed 100Mbps|
| --- | --- | --- |
| MediaPipe Runtime | | |
| Initial Page Load | 22.1KB | 0.04s |
| Initial Detector Creation: | | |
| Runtime | 1.57MB | |
| Lite model | 10.6MB | 1.91s |
| Full model | 14MB | 1.91s |
| Heavy model | 34.9MB | 4.82s |
| TensorFlow.js Runtime | | |
| Initial Page Load | 162.6KB | 0.07 |
| Initial Detector Creation: | | |
| Lite model | 10.41MB | 1.91s |
| Full model | 13.8MB | 1.91s |
| Heavy model | 34.7MB | 4.82s |
