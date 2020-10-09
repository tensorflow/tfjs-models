# Face landmarks detection

This project contains packages for detecting facial landmarks.

Currently, we offer one package: MediaPipe Facemesh (`mediapipe-facemesh`), described in detail below.

# MediaPipe Facemesh

MediaPipe Facemesh (`mediapipe-facemesh`) is a lightweight package predicting 486 3D facial landmarks to infer the approximate surface geometry of a human face ([paper](https://arxiv.org/pdf/1907.06724.pdf)).

<img src="demo.gif" alt="demo" style="width: 640px;"/>

More background information about the package, as well as its performance characteristics on different datasets, can be found in the model card: [https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view). The facemesh package optionally loads an iris detection model, whose model card can be found here: [https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view](https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view).

The package is designed for front-facing cameras on mobile devices, where faces in view tend to occupy a relatively large fraction of the canvas. MediaPipe Facemesh may struggle to identify far-away faces.

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/face-landmarks-detection/index.html), which uses the MediaPipe Facemesh to detect facial landmarks in a live video stream.

This package is also available as part of [MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/models), a
framework for building multimodal applied ML pipelines.

## Installation

Via script tags:

```html
<!-- Require the peer dependencies of face-landmarks-detection. -->
<script src="https://unpkg.com/@tensorflow/tfjs-core@2.4.0/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter@2.4.0/dist/tf-converter.js"></script>

<!-- You must explicitly require a TF.js backend if you're not using the tfjs union bundle. -->
<script src="https://unpkg.com/@tensorflow/tfjs-backend-webgl@2.4.0/dist/tf-backend-webgl.js"></script>
<!-- Alternatively you can use the WASM backend: <script src="https://unpkg.com/@tensorflow/tfjs-backend-wasm@2.4.0/dist/tf-backend-wasm.js"></script> -->

<!-- Require face-landmarks-detection itself. -->
<script src="https://unpkg.com/@tensorflow-models/face-landmarks-detection@0.0.1/dist/face-landmarks-detection.js"></script>
```

Via npm:

Using `yarn`:

    $ yarn add @tensorflow-models/face-landmarks-detection@0.0.1

    $ yarn add @tensorflow/tfjs-core@2.4.0, @tensorflow/tfjs-converter@2.4.0
    $ yarn add @tensorflow/tfjs-backend-webgl@2.4.0 # or @tensorflow/tfjs-backend-wasm@2.4.0

## Usage

If you are using via npm, first add:

```js
const faceLandmarksDetection = require('@tensorflow-models/face-landmarks-detection');

// If you are using the WebGL backend:
require('@tensorflow/tfjs-backend-webgl');

// If you are using the WASM backend:
// require('@tensorflow/tfjs-backend-wasm'); // You need to require the backend explicitly because facemesh itself does not
```

Then:

```js

async function main() {
  // Load the MediaPipe Facemesh package.
  const model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);

  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
  // array of detected faces from the MediaPipe graph. If passing in a video
  // stream, a single prediction per frame will be returned.
  const predictions = await model.estimateFaces({
    input: document.querySelector("video")
  });

  if (predictions.length > 0) {
    /*
    `predictions` is an array of objects describing each detected face, for example:

    [
      {
        faceInViewConfidence: 1, // The probability of a face being present.
        boundingBox: { // The bounding box surrounding the face.
          topLeft: [232.28, 145.26],
          bottomRight: [449.75, 308.36],
        },
        mesh: [ // The 3D coordinates of each facial landmark.
          [92.07, 119.49, -17.54],
          [91.97, 102.52, -30.54],
          ...
        ],
        scaledMesh: [ // The 3D coordinates of each facial landmark, normalized.
          [322.32, 297.58, -17.54],
          [322.18, 263.95, -30.54]
        ],
        annotations: { // Semantic groupings of the `scaledMesh` coordinates.
          silhouette: [
            [326.19, 124.72, -3.82],
            [351.06, 126.30, -3.00],
            ...
          ],
          ...
        }
      }
    ]
    */

    for (let i = 0; i < predictions.length; i++) {
      const keypoints = predictions[i].scaledMesh;

      // Log facial keypoints.
      for (let i = 0; i < keypoints.length; i++) {
        const [x, y, z] = keypoints[i];

        console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
      }
    }
  }
}

main();

```

#### Parameters for faceLandmarksDetection.load()

`faceLandmarksDetection.load()` takes two arguments:

* **package** - Which package to load. Current options: `mediapipe-facemesh`.

* **packageConfig** - A configuration object. If loading the `mediapipe-facemesh` package, the configuration object has the following properties:

  * **shouldLoadIrisModel** - Whether to load the MediaPipe iris detection model (an additional 2.6 MB of weights). The MediaPipe iris detection model provides (1) an additional 10 keypoints outlining the irises and (2) improved eye region keypoints enabling blink detection. Defaults to `true`.

  * **maxContinuousChecks** - How many frames to go without running the bounding box detector. Only relevant if maxFaces > 1. Defaults to 5.

  * **detectionConfidence** - Threshold for discarding a prediction. Defaults to 0.9.

  * **maxFaces** - The maximum number of faces detected in the input. Should be set to the minimum number for performance. Defaults to 10.

  * **iouThreshold** - A float representing the threshold for deciding whether boxes overlap too much in non-maximum suppression. Must be between [0, 1]. Defaults to 0.3. A score of 0 means no overlapping faces will be detected, whereas a score closer to 1 means the model will attempt to detect completely overlapping faces.

  * **scoreThreshold** - A threshold for deciding when to remove boxes based on score in non-maximum suppression. Defaults to 0.75. Increase this score in order to reduce false positives (detects fewer faces).

  * **modelUrl** - Optional param for specifying a custom facemesh model url or a `tf.io.IOHandler` object.

  * **irisModelUrl** - Optional param for specifying a custom iris model url or a `tf.io.IOHandler` object.

#### Parameters for model.estimateFaces()

* **config** - A configuration object. If loading the `mediapipe-facemesh` package, the configuration object has the following properties:

  * **input** - The image to classify. Can be a tensor, DOM element image, video, or canvas.

  * **returnTensors** - (defaults to `false`) Whether to return tensors as opposed to values.

  * **flipHorizontal** - Whether to flip/mirror the facial keypoints horizontally. Should be true for videos that are flipped by default (e.g. webcams).

  * **predictIrises** - (defaults to `true`) Whether to return keypoints for the irises. Disabling may improve performance.

#### Keypoints

Here is map of the keypoints:

<img src="mesh_map.jpg" alt="keypoints_map" style="width: 500px; height: 500px">

The UV coordinates for these keypoints are available via the `getUVCoords()` method on the `FaceLandmarksDetection` model object. They can also be found in `src/uv_coords.ts`.
