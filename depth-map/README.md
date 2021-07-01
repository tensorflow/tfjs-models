# DepthMap

DepthMap is a lightweight model capable of inferring depth from single images. It is implemented with [FastDepth](https://arxiv.org/abs/1903.03273).

![Input image - picture of a living room](https://github.com/tensorflow/tfjs-models/blob/master/depth-map/demo/livingroom.jpg) ![Resulting depth map](https://github.com/tensorflow/tfjs-models/blob/master/depth-map/demo/output.jpg)

# Performance

FastDepth consists of ~5MB of weights, and is well-suited for real time inference across a variety of devices.

TODO: Add performance

## Installation

Using `yarn`:

    $ yarn add @tensorflow-models/depth-map

Using `npm`:

    $ npm install @tensorflow-models/depth-map

Note that this package specifies `@tensorflow/tfjs-core` and `@tensorflow/tfjs-converter` as peer dependencies, so they will also need to be installed.

## Usage

To import in npm:
```js
const depthmap = require('@tensorflow-models/depth-map');
```

or as a standalone script tag:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/depth-map"></script>
```

Then:
```
const img = document.getElementById('img');

async function generateDepthMap() {
  // Load the model.
  const model = await depthmap.load();

  // The DepthMap model takes an image as input and returns a tensor containing the depth map for the image
  const output = await model.predict(img);
}
```
## API

### Loading the model
```ts
depthmap.load({modelUrl: string | tf.io.IOHandler,
    inputRange?: [number, number]): Promise<DepthMap>
```

Args:
- **modelUrl:** Optional param for specifying the custom model url or `tf.io.IOHandler` object.
- **inputRange:** Optional param specifying the pixel value range of your input. This is typically [0, 255] or [0, 1].
Defaults to [0, 255].

### Warming up the model (optional)

Models can take longer to run the first time, calling this method will allow future calls to predict to be faster.

```ts
depthmap.warmup()
```


### Making a depth prediction

You can make a prediction with DepthMap without needing to create a Tensor
with `depthmap.predict`, which takes an input image element and returns a
depth tensor.

This method exists on the model that is loaded from `depthmap.load`.

```ts
model.predict(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement,
): tf.Tensor2D
```

Returns a `Tensor2D` of shape [224, 224] where each float element corresponds to the estimated depth of that pixel in meters.

```
@inproceedings{icra_2019_fastdepth,
	author      = {{Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne}},
	title       = {{FastDepth: Fast Monocular Depth Estimation on Embedded Systems}},
	booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
	year        = {{2019}}
}
```
