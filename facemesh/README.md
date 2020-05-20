# MediaPipe Facemesh

MediaPipe Facemesh is a lightweight machine learning pipeline predicting 486 3D facial landmarks to infer the approximate surface geometry of a human face ([paper](https://arxiv.org/pdf/1907.06724.pdf)).

<img src="demo.gif" alt="demo" style="width: 640px;"/>

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view)

The model is designed for front-facing cameras on mobile devices, where faces in view tend to occupy a relatively large fraction of the canvas. MediaPipe Facemesh may struggle to identify far-away faces.

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/facemesh/index.html), which uses the model to detect facial landmarks in a live video stream.

This model is also available as part of [MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/models), a
framework for building multimodal applied ML pipelines.

## Installation

Using `yarn`:

    $ yarn add @tensorflow-models/facemesh

Using `npm`:

    $ npm install @tensorflow-models/facemesh

Note that this package specifies `@tensorflow/tfjs-core` and `@tensorflow/tfjs-converter` as peer dependencies, so they will also need to be installed.

## Usage

To import in npm:

```js
const facemesh = require('@tensorflow-models/facemesh');
```

or as a standalone script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/facemesh"></script>
```

Then:

```js

async function main() {
  // Load the MediaPipe facemesh model.
  const model = await facemesh.load();

  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
  // array of detected faces from the MediaPipe graph.
  const predictions = await model.estimateFaces(document.querySelector("video"));

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

#### Parameters for facemesh.load()

`facemesh.load()` takes a configuration object with the following properties:

* **maxContinuousChecks** - How many frames to go without running the bounding box detector. Only relevant if maxFaces > 1. Defaults to 5.

* **detectionConfidence** - Threshold for discarding a prediction. Defaults to 0.9.

* **maxFaces** - The maximum number of faces detected in the input. Should be set to the minimum number for performance. Defaults to 10.

* **iouThreshold** - A float representing the threshold for deciding whether boxes overlap too much in non-maximum suppression. Must be between [0, 1]. Defaults to 0.3.

* **scoreThreshold** - A threshold for deciding when to remove boxes based on score in non-maximum suppression. Defaults to 0.75.

#### Parameters for model.estimateFace()

* **input** - The image to classify. Can be a tensor, DOM element image, video, or canvas.

* **returnTensors** - (defaults to `false`) Whether to return tensors as opposed to values.

* **flipHorizontal** - Whether to flip/mirror the facial keypoints horizontally. Should be true for videos that are flipped by default (e.g. webcams).

#### Keypoints

Here is map of the keypoints:

<img src="mesh_map.jpg" alt="keypoints_map" style="width: 500px; height: 500px">

The UV coordinates for these keypoints are available via the `getUVCoords()` method on the `FaceMesh` model object. They can also be found in `src/uv_coords.ts`.
