# MPHands

MPHands TFJS uses TF.js runtime to execute the model, the preprocessing and postprocessing steps.

Note: this model can only detect a maximum of one hand in the input - multi-hand detection is coming in a future release.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

Via script tags:

```html
<!-- Require the peer dependencies of handpose. -->
<script src="https://unpkg.com/@tensorflow/tfjs-core@2.1.0/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter@2.1.0/dist/tf-converter.js"></script>

<!-- You must explicitly require a TF.js backend if you're not using the tfs union bundle. -->
<script src="https://unpkg.com/@tensorflow/tfjs-backend-webgl@2.1.0/dist/tf-backend-webgl.js"></script>
<!-- Alternatively you can use the WASM backend: <script src="https://unpkg.com/@tensorflow/tfjs-backend-wasm@2.1.0/dist/tf-backend-wasm.js"></script> -->

<script src="https://unpkg.com/@tensorflow-models/handpose@0.0.6/dist/handpose.js"></script>
```

Via npm:

Using `yarn`:

    $ yarn add @tensorflow-models/handpose
    $ yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
    $ yarn add @tensorflow/tfjs-backend-webgl # or @tensorflow/tfjs-backend-wasm

## Usage

If you are using npm, first add:

```js
const handpose = require('@tensorflow-models/handpose');

require('@tensorflow/tfjs-backend-webgl'); // handpose does not itself require a backend, so you must explicitly install one.

// If you are using the WASM backend:
// require('@tensorflow/tfjs-backend-wasm');
```

Then:

```js
async function main() {
  // Load the MediaPipe handpose model.
  const model = await handpose.load();
  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
  // hand prediction from the MediaPipe graph.
  const predictions = await model.estimateHands(document.querySelector("video"));
  if (predictions.length > 0) {
    /*
    `predictions` is an array of objects describing each detected hand, for example:
    [
      {
        handInViewConfidence: 1, // The probability of a hand being present.
        boundingBox: { // The bounding box surrounding the hand.
          topLeft: [162.91, -17.42],
          bottomRight: [548.56, 368.23],
        },
        landmarks: [ // The 3D coordinates of each hand landmark.
          [472.52, 298.59, 0.00],
          [412.80, 315.64, -6.18],
          ...
        ],
        annotations: { // Semantic groupings of the `landmarks` coordinates.
          thumb: [
            [412.80, 315.64, -6.18]
            [350.02, 298.38, -7.14],
            ...
          ],
          ...
        }
      }
    ]
    */

    for (let i = 0; i < predictions.length; i++) {
      const keypoints = predictions[i].landmarks;

      // Log hand keypoints.
      for (let i = 0; i < keypoints.length; i++) {
        const [x, y, z] = keypoints[i];
        console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
      }
    }
  }
}
main();
```

#### Parameters for handpose.load()

`handpose.load()` takes a configuration object with the following properties:

* **maxContinuousChecks** - How many frames to go without running the bounding box detector. Defaults to infinity. Set to a lower value if you want a safety net in case the mesh detector produces consistently flawed predictions.

* **detectionConfidence** - Threshold for discarding a prediction. Defaults to 0.8.

* **iouThreshold** - A float representing the threshold for deciding whether boxes overlap too much in non-maximum suppression. Must be between [0, 1]. Defaults to 0.3.

* **scoreThreshold** - A threshold for deciding when to remove boxes based on score in non-maximum suppression. Defaults to 0.75.

#### Parameters for handpose.estimateHands()

* **input** - The image to classify. Can be a tensor, DOM element image, video, or canvas.

* **flipHorizontal** - Whether to flip/mirror the hand keypoints horizontally. Should be true for videos that are flipped by default (e.g. webcams).
