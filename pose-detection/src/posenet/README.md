# PoseNet

PoseNet can be used to estimate either a single pose or multiple poses, meaning
there is a version of the algorithm that can detect only one person in an image/video
and one version that can detect multiple persons in an image/video.

[Refer to this blog post](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) for a
high-level description of PoseNet running on Tensorflow.js. Please try it out using the live
[demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=posenet).

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

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

-------------------------------------------------------------------------------

## Usage

If you are using the Pose API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
```

### Create a detector

Pass in `poseDetection.SupportedModels.PoseNet` from the
`posedetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines PoseNet specific configurations for `PosenetModelConfig`:

* **architecture**: Optional. Can be either `MobileNetV1` or `ResNet50`. It determines which PoseNet architecture to load.
  Defaults to `MobileNetV1`.

* **outputStride**: Optional. Can be one of `8`, `16`, `32` (Stride `16`, `32` are supported for the ResNet architecture and stride `8`, `16`, `32` are supported for the MobileNetV1 architecture. However if you are using stride `32` you must set the multiplier to `1.0`). It specifies the output stride of the PoseNet model. The smaller the value, the larger the output resolution, and more accurate the model at the cost of speed. Set this to a larger value to increase speed at the cost of accuracy. Defaults to 16.

* **inputResolution**: Optional. A `number` or an `Object` of type `{width: number, height: number}`. Defaults to `257.` It specifies the size the image is resized and padded to before it is fed into the PoseNet model. The larger the value, the more accurate the model at the cost of speed. Set this to a smaller value to increase speed at the cost of accuracy. If a number is provided, the image will be resized and padded to be a square with the same width and height.  If `width` and `height` are provided, the image will be resized and padded to the specified width and height.

* **multiplier**: Optional.Can be one of `1.0`, `0.75`, or `0.50` (The value is used *only* by the MobileNetV1 architecture and not by the ResNet architecture). It is the float multiplier for the depth (number of channels) for all convolution ops. The larger the value, the larger the size of the layers, and more accurate the model at the cost of speed. Set this to a smaller value to increase speed at the cost of accuracy. Defaults to 1.0.

* **quantBytes**: Optional. This argument controls the bytes used for weight quantization. The available options are:

   - `4`: 4 bytes per float (no quantization). Leads to highest accuracy and original model size (~90MB).
   - `2`: 2 bytes per float. Leads to slightly lower accuracy and 2x model size reduction (~45MB).
   - `1`: 1 byte per float. Leads to lower accuracy and 4x model size reduction (~22MB).

  Defaults to 4.

* **modelUrl**: Optional. A string that specifies custom url of the model. This is useful for local development or countries that don't have access to the model hosted on GCP.

The following code snippet demonstrates how to load the model:

*MobileNet (smaller, faster, less accurate)*
```javascript
const detectorConfig = {
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: { width: 640, height: 480 },
  multiplier: 0.75
};
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet, detectorConfig);
```

*ResNet (larger, slower, more accurate) \*\*new!\*\**
```javascript
const detectorConfig = {
  architecture: 'ResNet50',
  outputStride: 32,
  inputResolution: { width: 257, height: 200 },
  quantBytes: 2
};
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet, detectorConfig);
```

### Run inference

Now you can use the detector to detect poses. The `estimatePoses` method
accepts both image and video in many formats, including:
`tf.Tensor3D`, `ImageData`, `HTMLVideoElement`, `HTMLImageElement`,
`HTMLCanvasElement`. If you want more options, you can pass in a second
`estimationConfig` parameter.

`estimationConfig` is an object that defines BlazePose specific configurations for
`PoseNetEstimationConfig`:

*   *maxPoses*: Optional. Max number of poses to detect. Defaults to 1, which means
    single pose detection. Single pose detection runs more efficiently, while
    multi-pose (maxPoses > 1) detection is usually much slower. Multi-pose
    detection should only be used when needed.

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from
    camera, the result has to flip horizontally.

*   *scoreThreshold*: Optional. For maxPoses > 1. Only return instance detections that have
    root part score greater or equal to this value. Defaults to 0.5

*   *nmsRadius*: Optional. For maxPoses > 1. Non-maximum suppression part distance in
    pixels. It needs to be strictly positive. Two parts suppress each other if
    they are less than `nmsRadius` pixels away. Defaults to 20.

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {
  maxPoses: 5,
  flipHorizontal: false,
  scoreThreshold: 0.5,
  nmsRadius: 20
};
const poses = await detector.estimatePoses(image, estimationConfig);
```

Please refer to the Pose API
[README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation)
about the structure of the returned `poses`.
