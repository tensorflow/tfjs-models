# MediaPipeFaceMesh

MediaPipeFaceMesh-TFJS uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.

Please try our our live [demo](https://storage.googleapis.com/tfjs-models/demos/face-landmarks-detection/index.html?model=mediapipe_face_mesh).
In the runtime-backend dropdown, choose 'tfjs-webgl'.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use MediaPipeFaceMesh, you need to first select a runtime (TensorFlow.js or MediaPipe).
This guide is for TensorFlow.js
runtime. The guide for MediaPipe runtime can be found
[here](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection/src/mediapipe).

Via script tags:

```html
<!-- Require the peer dependencies of face-landmarks-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
yarn add @tensorflow-models/face-detection
yarn add @tensorflow-models/face-landmarks-detection
```

-----------------------------------------------------------------------
## Usage

If you are using the face-landmarks-detection API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import '@mediapipe/face_mesh';
```
### Create a detector

Pass in `faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh` from the
`faceLandmarksDetection.SupportedModels` enum list along with a `detectorConfig` to the
`createDetector` method to load and initialize the model.

`detectorConfig` is an object that defines MediaPipeFaceMesh specific configurations for `MediaPipeFaceMeshTfjsModelConfig`:

*   *runtime*: Must set to be 'tfjs'.

*   *maxFaces*: Defaults to 1. The maximum number of faces that will be detected by the model. The number of returned faces can be less than the maximum (for example when no faces are present in the input). It is highly recommended to set this value to the expected max number of faces, otherwise the model will continue to search for the missing faces which can slow down the performance.

*   *refineLandmarks*: Defaults to false. If set to true, refines the landmark coordinates around the eyes and lips, and output additional landmarks around the irises.

*   *detectorModelUrl*: An optional string that specifies custom url of
the detector model. This is useful for area/countries that don't have access to the model hosted on tf.hub. It also accepts `io.IOHandler` which can be used with
[tfjs-react-native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
to load model from app bundle directory using
[bundleResourceIO](https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/src/bundle_resource_io.ts#L169).
*   *landmarkModelUrl* An optional string that specifies custom url of
the landmark model. This is useful for area/countries that don't have access to the model hosted on tf.hub. It also accepts `io.IOHandler` which can be used with
[tfjs-react-native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
to load model from app bundle directory using
[bundleResourceIO](https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/src/bundle_resource_io.ts#L169).

```javascript
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
  runtime: 'tfjs',
};
detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
```

### Run inference

Now you can use the detector to detect faces. The `estimateFaces` method
accepts both image and video in many formats, including: `tf.Tensor3D`,
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement` and `Tensor3D`. If you want more
options, you can pass in a second `estimationConfig` parameter.

`estimationConfig` is an object that defines MediaPipeFaceMesh specific configurations for `MediaPipeFaceMeshTfjsEstimationConfig`:

*   *flipHorizontal*: Optional. Defaults to false. When image data comes from camera, the result has to flip horizontally.

*   *staticImageMode*: Optional. Defaults to false. If set to true, face detection
will run on every input image, otherwise if set to false then detection runs
once and then the model simply tracks those landmarks without invoking
another detection until it loses track of any of the faces (ideal for videos).

The following code snippet demonstrates how to run the model inference:

```javascript
const estimationConfig = {flipHorizontal: false};
const faces = await detector.estimateFaces(image, estimationConfig);
```

Please refer to the Face API
[README](https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/README.md#how-to-run-it)
about the structure of the returned `faces` array.
