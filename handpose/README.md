# MediaPipe Handpose

MediaPipe Handpose is a model for predicting 3D hand keypoints.

<img src="demo/demo.gif" alt="demo" style="width:640px" />

Given an input, the model predicts whether it contains a hand. If so, the model returns coordinates for the bounding box around the hand, as well as 21 keypoints within the hand, outlining the location of each finger joint and the palm.

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1sv4sSb9BSNVZhLzxXJ0jBv9DqD-4jnAz/view](https://drive.google.com/file/d/1sv4sSb9BSNVZhLzxXJ0jBv9DqD-4jnAz/view)

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/handpose/index.html), which uses the model to detect hand landmarks in a live video stream.

This model is also available as part of [MediaPipe](https://hand.mediapipe.dev/), a framework for building multimodal applied ML pipelines.

# Performance

MediaPipe Handpose consists of ~12MB of weights, and is well-suited for real time inference across a variety of devices (40 FPS on a 2018 MacBook Pro, 9 FPS on an iPhone6).

## Installation

Using `yarn`:

    $ yarn add @tensorflow-models/handpose

Using `npm`:

    $ npm install @tensorflow-models/handpose

Note that this package specifies `@tensorflow/tfjs-core` and `@tensorflow/tfjs-converter` as peer dependencies, so they will also need to be installed.

## Usage

To import in npm:

```js
import * as handpose from '@tensorflow-models/handpose';
```

or as a standalone script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/handpose"></script>
```

Then:

```js
async function main() {
  // Load the MediaPipe handpose model.
  const model = await handpose.load();
  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
  // hand prediction from the MediaPipe graph.
  const prediction = await model.estimateHand(document.querySelector("video"));
  if (prediction != null) {
    /*
    `prediction` is an object describing the detected hand, for example:
    [
      {
        handInViewConfidence: 1, // The probability of a face being present.
        boundingBox: { // The bounding box surrounding the face.
          topLeft: [232.28, 145.26],
          bottomRight: [449.75, 308.36],
        },
        landmarks: [ // The 3D coordinates of each facial landmark.
          [92.07, 119.49, -17.54],
          [91.97, 102.52, -30.54],
          ...
        ]
      }
    ]
    */

    const keypoints = predictions[i].scaledMesh;
    // Log facial keypoints.
    for (let i = 0; i < keypoints.length; i++) {
      const [x, y, z] = keypoints[i];
      console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
    }
  }
}
main();
```
