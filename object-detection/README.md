# Object Detection (coco-ssd)

Object detection model aims to localize and identify multiple objects in a single image.

This TensorFlow.js object detection model is built on top of Tensorflow object detection API. For more information about Tensorflow object detection API, check out this readme in
[tensorflow/object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).

This TensorFlow.js model does not require you to know about machine learning.
It can take as input any browser-based image elements (`<img>`, `<video>`, `<canvas>`
elements, for example) and returns an array of most bounding boxes with class name and confidence level.

## Usage

There are two main ways to get this model in your JavaScript project: via script tags or by installing it from NPM and using a build tool like Parcel, WebPack, or Rollup.

### via Script Tag

```html
<!-- Load TensorFlow.js. This is required to use MobileNet. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.12.6"> </script>
<!-- Load the MobileNet model. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/object-detection@0.1.0"> </script>

<!-- Replace this with your image. Make sure CORS settings allow reading the image! -->
<img id="img" src="cat.jpg"/>

<!-- Place your code in the script tag below. You can also use an external .js file -->
<script>
  // Notice there is no 'import' statement. 'mobilenet' and 'tf' is
  // available on the index-page because of the script tag above.

  const img = document.getElementById('img');

  // Load the model.
  objectDetection.load().then(model => {
    // Classify the image.
    model.detect(img).then(predictions => {
      console.log('Predictions: ', predictions);
    });
  });
</script>
```

### via NPM

```js
// Note: you do not need to import @tensorflow/tfjs here.

import * as objectDetection from '@tensorflow-models/object-detection';

const img = document.getElementById('img');

// Load the model.
const model = await objectDetection.load();

// Classify the image.
const predictions = await model.detect(img);

console.log('Predictions: ');
console.log(predictions);
```

You can also take a look at the [demo app](./demo).

## API

#### Loading the model
`object-detection` is the module name, which is automatically included when you use the `<script src>` method. When using ES6 imports, object-detection is the module.

```ts
objectDetection.load(
  base?: 'ssd_mobilenet_v1' | 'ssd_mobilenet_v2' | 'ssdlite_mobilenet_v2'
)
```

Args:
 **base:** Controls the base cnn model, can be 'ssd_mobilenet_v1', 'ssd_mobilenet_v2' or 'ssdlite_mobilenet_v2'. Defaults to 'ssdlite_mobilenet_v2'.

Returns a `model` object.

#### Detecting the objects

You can detect objects with the model without needing to create a Tensor.
`model.detect` takes an input image element and returns an array of bounding boxes with class name and confidence level.

This method exists on the model that is loaded from `objectDetection.load`.

```ts
model.detect(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement, maxDetectionSize: number
)
```

Args:
- **img:** A Tensor or an image element to make a detection on.
- **maxDetectionSize:** The max count of detected objects, default to 20.

Returns an array of classes and probabilities that looks like:

```js
[{
  bbox: [x, y, width, height],
  class: "person",
  score: 0.8380282521247864
}, {
  bbox: [x, y, width, height],
  class: "kite",
  score: 0.74644153267145157
}]
```

### Preparing the model

This model is based on the TensorFlow object detection API with following some optimizations:

  1. Removed the post process graph from the original model.
  2. Used single class NonMaxSuppression instead of original multiple classes NonMaxSuppression for faster speed with similar accuracy.
  3. Executes NonMaxSuppression operations on CPU backend instead of WebGL to greatly improve the performance by avoiding the texture downloads.

Here is the converter command for removing the post process graph.

```
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='Postprocessor/ExpandDims_1,Postprocessor/Slice' --saved_model_tags=serve ./saved_model ./web_model
```