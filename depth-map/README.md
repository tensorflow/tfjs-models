# FastDepth

DepthMap is a lightweight model capable of infering depth from single images. It is implemented with [FastDepth](https://arxiv.org/abs/1903.03273).

# Performance

FastDepth consists of ~5MB of weights, and is well-suited for real time inference across a variety of devices.

TODO: Add performance

## Installation

And then in your project:

Using `yarn`:

    $ yarn add @tensorflow-models/fast-depth

Using `npm`:

    $ npm install @tensorflow-models/fast-depth

Note that this package specifies `@tensorflow/tfjs-core` and `@tensorflow/tfjs-converter` as peer dependencies, so they will also need to be installed.

## Usage

To import in npm:
```js
const depthmap = require('@tensorflow-models/depth-map');

const img = document.getElementById('img');

// Load the model.
const model = await depthmap.load({modelUrl: 'path/to/model.json'});

// Generate the depth map
const output = await model.predict(img);
```
## API

### Loading the model
```ts
depthmap.load({modelUrl: string | tf.io.IOHandler,
    inputRange?: [number, number],
    rawOutput?: boolean}): Promise<DepthMap>
```

Args:
- **modelUrl:** Param for specifying the custom model url or `tf.io.IOHandler` object.
Returns a `model` object.
- **inputRange:** Optional param specifying the pixel value range of your input. This is typically [0, 255] or [0, 1].
Defaults to [0, 255].
- **rawOutput:** Optional param specifying whether model shoudl output the raw [3, 224, 224] result or postprocess to
the image-friendly [224, 224, 3]. Defaults to false.


#### Making a depth prediction

You can make a prediciton with DepthMap without needing to create a Tensor
with `depthmap.predict`, which takes an input image element and returns a
depth tensor.

This method exists on the model that is loaded from `depthmap.load`.

```ts
model.predict(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
): tf.Tensor
```
