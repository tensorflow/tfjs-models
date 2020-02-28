# Facemesh

Facemesh is lightweight model that predicts 3D facial keypoints ([paper](https://arxiv.org/pdf/1907.06724.pdf)).

<img src="demo/demo.gif" alt="demo" style="width: 640px;"/>

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view)

The model is designed for front-facing cameras on mobile devices, where faces in view tend to occupy a relatively large fraction of the canvas. Facemesh may struggle to identify far-away faces.

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
import * as facemesh from '@tensorflow-models/facemesh';
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
  // Load the model.
  const model = await facemesh.load();

  // Pass in a video stream to obtain an array of detected faces.
  const predictions = await model.estimateFaces(document.querySelector("video"));

  if (predictions.length > 0) {
    /*
    `predictions` is an array of objects describing each detected face, for example:

    [
      {
        faceInViewConfidence: 1,
        boundingBox: {
          topLeft: [232.28, 145.26],
          bottomRight: [449.75, 308.36],
        },
        mesh: [
          [92.07, 119.49, -17.54],
          [91.97, 102.52, -30.54],
          ...
        ],
        scaledMesh: [
          [322.32, 297.58, -17.54],
          [322.18, 263.95, -30.54]
        ],
        annotations: {
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

      // Draw a circle over each facial keypoint.
      for (let i = 0; i < keypoints.length; i++) {
        const x = keypoints[i][0];
        const y = keypoints[i][1];
        ctx.beginPath();
        ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

main();

```


