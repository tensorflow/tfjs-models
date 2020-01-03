# Blazeface detector

[Blazeface](https://arxiv.org/abs/1907.05047) is a lightweight model that detects faces in images. Blazeface makes use of the [Single Shot Detector](https://arxiv.org/abs/1512.02325) architecture with a custom encoder. The model may serve as a first step for face-related computer vision applications, such as facial keypoint recognition.

<img src="demo/demo.gif" alt="demo" style="width: 437px;"/>

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view](https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view)

The model is designed for front-facing cameras on mobile devices, where faces in view tend to occupy a relatively large fraction of the canvas. Blazeface may struggle to identify far-away faces.

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/blazeface/index.html), which uses the model to predict facial bounding boxes from a live video stream.

This model is also available as part of
[MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/models), a
framework for building multimodal applied ML pipelines.

## Installation

Using `yarn`:

    $ yarn add @tensorflow-models/blazeface

Using `npm`:

    $ npm install @tensorflow-models/blazeface

Note that this package specifies `@tensorflow/tfjs-core` and `@tensorflow/tfjs-converter` as peer dependencies, so they will also need to be installed.

## Usage

To import in npm:

```js
import * as blazeface from '@tensorflow-models/blazeface';
```

or as a standalone script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface"></script>
```

Then:

```js

async function main() {
  // Load the model.
  const model = await blazeface.load();

  // Pass in an image or video to the model. The model returns an array of
  // bounding boxes, probabilities, and landmarks, one for each detected face.

  const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
  const predictions = await model.estimateFaces(document.querySelector("img"), returnTensors);

  if (predictions.length > 0) {
    /*
    `predictions` is an array of objects describing each detected face, for example:

    [
      {
        topLeft: [232.28, 145.26],
        bottomRight: [449.75, 308.36],
        probability: [0.998],
        landmarks: [
          [295.13, 177.64], // right eye
          [382.32, 175.56], // left eye
          [341.18, 205.03], // nose
          [345.12, 250.61], // mouth
          [252.76, 211.37], // right ear
          [431.20, 204.93] // left ear
        ]
      }
    ]
    */

    for (let i = 0; i < predictions.length; i++) {
      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      // Render a rectangle over each detected face.
      ctx.fillRect(start[0], start[1], size[0], size[1]);
    }
  }
}

main();

```
