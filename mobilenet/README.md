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
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.1"> </script>
<!-- Load the MobileNet model. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0"> </script>

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
`mobilenet` is the module name, which is automatically included when you use the `<script src>` method. When using ES6 imports, mobilenet is the module.

```ts
mobilenet.load({
    version: 1,
    alpha?: 0.25 | .50 | .75 | 1.0,
    modelUrl?: string
    inputRange?: [number, number]
  }
)
```
___For users of previous versions (1.0.x), the API is:___

```ts
mobilenet.load(
    version?: 1,
    alpha?: 0.25 | .50 | .75 | 1.0
)
```



Args:
- **version:** The MobileNet version number. Use 1 for [MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), and 2 for [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet). Defaults to 1.
- **alpha:** Controls the width of the network, trading accuracy for performance. A smaller alpha decreases accuracy and increases performance. 0.25 is only available for V1. Defaults to 1.0.
- **modelUrl:** Optional param for specifying the custom model url or `tf.io.IOHandler` object.
Returns a `model` object.
- **inputRange:** Optional param specifying the pixel value range expected by the trained model hosted at the modelUrl. This is typically [0, 1] or [-1, 1].

`mobilenet` is the module name, which is automatically included when you use
the <script src> method. When using ES6 imports, mobilenet is the module.


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

Returns a Promise that resolves to an array of classes and probabilities that looks like:

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

#### Getting embeddings

You can also get the embedding of an image to do transfer learning. The size
of the embedding depends on the alpha (width) of the model.

```ts
model.infer(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
  embedding = false
)
```

- **img:** A Tensor or an image element to make a classification on.
- **embedding:** If true, it returns the embedding. Otherwise it returns the 1000-dim unnormalized logits.
