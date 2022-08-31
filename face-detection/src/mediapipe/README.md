# MediaPipeFaceDetector

MediaPipeFaceDetector-MediaPipe wraps the MediaPipe JS Solution within the familiar
TFJS API [mediapipe.dev](https://mediapipe.dev).

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/face-detection/index.html?model=mediapipe_face_detector).

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use MediaPipeFaceDetector:

Via script tags:

```html
<!-- Require the peer dependencies of face-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection"></script>
```

Via npm:
```sh
yarn add @mediapipe/face_detection
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-backend-webgl
yarn add @tensorflow-models/face-detection
```

-----------------------------------------------------------------------
## Usage

If you are using the face-detection API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import '@mediapipe/face_detection';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';
```

### Create a detector

Pass in `faceDetection.SupportedModels.MediaPipeFaceDetector` from the
`faceDetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines MediaPipeFaceDetector specific configurations for `MediaPipeFaceDetectorMediaPipeModelConfig`:

*   *runtime*: Must set to be 'mediapipe'.

*   *maxFaces*: Defaults to 1. The maximum number of faces that will be detected by the model. The number of returned faces can be less than the maximum (for example when no faces are present in the input). It is highly recommended to set this value to the expected max number of faces, otherwise the model will continue to search for the missing faces which can slow down the performance.

*   *modelType*: Optional. Possible values: 'short'|'full'. Defaults to 'short'. The short-range model that works best for faces within 2 meters from the camera, while the full-range model works best for faces within 5 meters. For the full-range option, a sparse model is used for its improved inference speed.

*   *solutionPath*: The path to where the wasm binary and model files are located.

```javascript
const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
const detectorConfig = {
  runtime: 'mediapipe',
  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection',
                // or 'base/node_modules/@mediapipe/face_detection' in npm.
};
detector = await faceDetection.createDetector(model, detectorConfig);
```

### Run inference

Now you can use the detector to detect faces. The `estimateFaces` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement` and `Tensor3D`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is an object that defines MediaPipeFaceDetector specific configurations for `MediaPipeFaceDetectorMediaPipeEstimationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {flipHorizontal: false};
const faces = await detector.estimateFaces(image, estimationConfig);
```

Please refer to the Face API
[README](https://github.com/tensorflow/tfjs-models/blob/master/face-detection/README.md#how-to-run-it)
about the structure of the returned `faces` array.
