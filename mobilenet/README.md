# MobileNet

MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used.

MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.

This TensorFlow.js model does not require you to know about machine learning.
It can take as input any browser-based image elements (`<img>`, `<video>`, `<canvas>`
elements, for example) and returns an array of most likely predictions and
their confidences.

For more information about MobileNet, check out this readme in
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

## Usage

There are two main ways to get this model in your JavaScript project: via script tags or by installing it from NPM and using a build tool like Parcel, WebPack, or Rollup.

### via Script Tag

```html
<!-- Load TensorFlow.js. This is required to use MobileNet. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7"> </script>
<!-- Load the MobileNet model. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@0.1.1"> </script>

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
```

### via NPM

```js
// Note: you do not need to import @tensorflow/tfjs here.

import * as mobilenet from '@tensorflow-models/mobilenet';

const img = document.getElementById('img');

// Load the model.
const model = await mobilenet.load();

// Classify the image.
const predictions = await model.classify(img);

console.log('Predictions: ');
console.log(predictions);
```

## API

#### Loading the model
`mobilenet` is the module name, which is automatically included when you use
the <script src> method. When using ES6 imports, mobilenet is the module.

```ts
mobilenet.load(
  version?: 1,
  alpha?: 0.25 | .50 | .75 | 1.0
)
```

Args:
- **version:** The MobileNet version number. Currently only accepts and defaults to version 1. In the future we will support MobileNet V2.
- **alpha:** Controls the width of the network, trading accuracy for performance. A smaller alpha decreases accuracy and increases performance. Defaults to 1.0.

Returns a `model` object.

#### Making a classification

You can make a classification with mobilenet without needing to create a Tensor
with `MobileNet.classify`, which takes an input image element and returns an
array with top classes and their probabilities.

If you want to use this for transfer learning, see the `infer` method.

This method exists on the model that is loaded from `mobilenet.load`.

```ts
model.classify(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
  topk?: number
)
```

Args:
- **img:** A Tensor or an image element to make a classification on.
- **topk:** How many of the top probabilities to return. Defaults to 3.

Returns an array of classes and probabilities that looks like:

```js
[{
  className: "Egyptian cat",
  probability: 0.8380282521247864
}, {
  className: "tabby, tabby cat",
  probability: 0.04644153267145157
}, {
  className: "Siamese cat, Siamese",
  probability: 0.024488523602485657
}]
```

#### Getting activations

You can also use this model to get intermediate activations or logits as
TensorFlow.js tensors.

This method exists on the model that is loaded from `mobilenet.load`.

```ts
model.infer(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
  endpoint?: string
)
```

- **img:** A Tensor or an image element to make a classification on.
- **endpoint:** The optional endpoint to predict through. You can list all the endpoints with `model.endpoints`. These correspond to layers of the MobileNet model. If undefined, will return 1000D unnormalized logits.
