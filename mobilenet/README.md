# MobileNet

MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used.

MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.

This TensorFlow.js model does not require you to know about machine learning.
It can take as input any browser-based image elements (<img>, <video>, <canvas>
elements, for example) and returns an array of most likely predictions and
their confidences.

For more information about MobileNet, check out this readme in
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

## Usage

There are two main ways to get this model in your JavaScript project: via script tags or by installing it from NPM and using a build tool like Parcel, WebPack, or Rollup.

### via Script Tag

```html
<html>
  <head>
    <!-- Load TensorFlow.js. This is required to use MobileNet. -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.10.3"> </script>
    <!-- Load the MobileNet model. -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@0.1.0"> </script>

    <!-- Replace this with your image. Make sure CORS settings allow reading the image! -->
    <img id="img" src="cat.jpg"></img>

    <!-- Place your code in the script tag below. You can also use an external .js file -->
    <script>
      // Notice there is no 'import' statement. 'mobilenet' and 'tf' is
      // available on the index-page because of the script tag above.

      const img = document.getElementById('img');

      // Load the model.
      mobilenet.load().then(model => {
        // Classify the image.
        model.classify(img).then(predictions => {
          console.log('Predictions: ');
          console.log(predictions);
        });
      });
    </script>
  </head>

  <body>
  </body>
</html>
```

### via NPM

```js
// Note: you do not need to import @tensorflow/tfjs here.

import * as mobilenet from '@tensorflow-models/mobilenet';

const img = document.getElementById('img');

// Load the model.
mobilenet.load().then(model => {
  // Classify the image.
  model.classify(img).then(predictions => {
    console.log('Predictions: ');
    console.log(predictions);
  });
});
```

## API

#### Loading the model
```ts
mobilenet.load(
  version?: 1,
  alpha?: 0.25 | .50 | .75 | 1.0)
```

Args:
- version: The MobileNet version number. Currently only accepts and defaults to version 1. In the future we will support MobileNet V2.
- alpha: Controls the width of the network, trading accuracy for performance. A smaller alpha decreases accuracy and increases performance. Defaults to 1.0.

Returns a `model` object.

#### Making a classification
```ts
model.classify(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
  topk?: number
)

Args:
- img: A Tensor or an image element to make a classification on.
- topk: How many of the top probabilities to return. Defaults to 3.
