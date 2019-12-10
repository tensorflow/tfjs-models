# Blazeface detector

[Blazeface](https://arxiv.org/abs/1907.05047) is a lightweight model that detects faces in images. Blazeface makes use of the [Single Shot Detector](https://arxiv.org/abs/1512.02325) architecture with a custom encoder. The model may serve as a first step for face-related computer vision applications, such as facial keypoint recognition.

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view](https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view)

The model is designed for front-facing cameras on mobile devices, where faces in view tend to occupy a relatively large fraction of the canvas. Blazeface may struggle to identify far-away faces.

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/blazeface/index.html), which uses the model to predict facial bounding boxes from a live video stream.

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

// Load the model.
const model = await blazeface.load();

// Pass in an image or video to the model. The model returns an array of
// bounding boxes, one for each detected face.
const prediction = await model.estimateFace(document.querySelector("img"));

if (prediction) {
  for (let i = 0; i < prediction.length; i++) {

    // The first element of each bounding box specifies the upper left hand
    // corner of the detected face. The second element specifies the lower right
    // hand corner.
    const start = prediction[i][0];
    const end = prediction[i][1];
    const size = [end[0] - start[0], end[1] - start[1]];

    ctx.fillRect(start[0], start[1], size[0], size[1]);
  }
}

```
