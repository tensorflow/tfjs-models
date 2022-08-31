# Semantic Segmentation in the Browser: DeepLab v3 Model

This package contains a standalone implementation of the DeepLab inference pipeline, as well as a [demo](./demo), for running semantic segmentation using TensorFlow.js.

![DeepLab Demo](./docs/deeplab-demo.gif)

## Usage

In the first step of semantic segmentation, an image is fed through a pre-trained model [based](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) on MobileNet-v2. Three types of pre-trained weights are available, trained on [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [Cityscapes](https://www.cityscapes-dataset.com) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) datasets.

To get started, pick the model name from `pascal`, `cityscapes` and `ade20k`, and decide whether you want your model quantized to 1 or 2 bytes (set the `quantizationBytes` option to 4 if you want to disable quantization). Then, initialize the model as follows:

```typescript
const tf = require('@tensorflow-models/tfjs');
const deeplab = require('@tensorflow-models/deeplab');
const loadModel = async () => {
  const modelName = 'pascal';   // set to your preferred model, either `pascal`, `cityscapes` or `ade20k`
  const quantizationBytes = 2;  // either 1, 2 or 4
  return await deeplab.load({base: modelName, quantizationBytes});
};

const input = tf.zeros([227, 500, 3]);
// ...

loadModel()
    .then((model) => model.segment(input))
    .then(
        ({legend}) =>
            console.log(`The predicted classes are ${JSON.stringify(legend)}`));
```

By default, calling `load` initalizes the PASCAL variant of the model quantized to 2 bytes.

If you would rather load custom weights, you can pass the URL in the config instead:

```typescript
const deeplab = require('@tensorflow-models/deeplab');
const loadModel = async () => {
  const url = 'https://tfhub.dev/tensorflow/tfjs-model/deeplab/pascal/1/default/1/model.json?tfjs-format=file';
  return await deeplab.load({modelUrl: url});
};
loadModel().then(() => console.log(`Loaded the model successfully!`));
```

This will initialize and return the `SemanticSegmentation` model.

You can set the `base` attribute in the argument to `pascal`, `cityscapes` or `ade20k` to use the corresponding colormap and labelling scheme. Otherwise, you would have to provide those yourself during segmentation.

If you require more careful control over the initialization and behavior of the model (e.g. you want to use your own labelling scheme and colormap), use the `SemanticSegmentation` class, passing a pre-loaded `GraphModel` in the constructor:

```typescript
const tfconv = require('@tensorflow/tfjs-converter');
const deeplab = require('@tensorflow-models/deeplab');
const loadModel = async () => {
  const base = 'pascal';        // set to your preferred model, out of `pascal`,
                                // `cityscapes` and `ade20k`
  const quantizationBytes = 2;  // either 1, 2 or 4
  // use the getURL utility function to get the URL to the pre-trained weights
  const modelUrl = deeplab.getURL(base, quantizationBytes);
  const rawModel = await tfconv.loadGraphModel(modelUrl);
  const modelName = 'pascal';  // set to your preferred model, out of `pascal`,
  // `cityscapes` and `ade20k`
  return new deeplab.SemanticSegmentation(rawModel);
};
loadModel().then(() => console.log(`Loaded the model successfully!`));
```

Use `getColormap(base)` and `getLabels(base)` utility function to fetch the default colormap and labelling scheme.

```typescript
import {getLabels, getColormap} from '@tensorflow-models/deeplab';
const model = 'ade20k';
const colormap = getColormap(model);
const labels = getLabels(model);
```

### Segmenting an Image

The `segment` method of the `SemanticSegmentation` object covers most use cases.

Each model recognises a different set of object classes in an image:

- [PASCAL](./src/config.ts#L142)
- [CityScapes](./src/config.ts#L149)
- [ADE20K](./src/config.ts#L155)

#### `model.segment(image, config?)` inputs

- **image** :: `ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D`;

  The image to segment

- **config.canvas** (optional) :: `HTMLCanvasElement`

  Pass an optional canvas element as `canvas` to draw the output

- **config.colormap** (optional) :: `[number, number, number][]`

  The array of RGB colors corresponding to labels

- **config.labels** (optional) :: `string[]`

  The array of names corresponding to labels

  By [default](./src/index.ts#L81), `colormap` and `labels` are set according to the `base` model attribute passed during initialization.

#### `model.segment(image, config?)` outputs

The output is a promise of a `DeepLabOutput` object, with four attributes:

- **legend** :: `{ [name: string]: [number, number, number] }`

  The legend is a dictionary of objects recognized in the image and their colors in RGB format.

- **height** :: `number`

  The height of the returned segmentation map

- **width** :: `number`

  The width of the returned segmentation map

- **segmentationMap** :: `Uint8ClampedArray`

  The colored segmentation map as `Uint8ClampedArray` which can be [fed](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Pixel_manipulation_with_canvas) into `ImageData` and mapped to a canvas.

#### `model.segment(image, config?)` example

```typescript
const classify = async (image) => {
    return await model.segment(image);
}
```

**Note**: *For more granular control, consider `predict` and `toSegmentationImage` methods described below.*

### Producing a Semantic Segmentation Map

To segment an arbitrary image and generate a two-dimensional tensor with class labels assigned to each cell of the grid overlayed on the image (with the maximum number of cells on the side fixed to 513), use the `predict` method of the `SemanticSegmentation` object.

#### `model.predict(image)` input

- **image** :: `ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D`;

  The image to segment

#### `model.predict(image)` output

- **rawSegmentationMap** :: `tf.Tensor2D`

  The segmentation map of the image

#### `model.predict(image)` example

```javascript
const getSemanticSegmentationMap = (image) => {
    return model.predict(image)
}
```

### Translating a Segmentation Map into the Color-Labelled Image

To transform the segmentation map into a coloured image, use the `toSegmentationImage` method.

#### `toSegmentationImage(colormap, labels, segmentationMap, canvas?)` inputs

- **colormap** :: `[number, number, number][]`

  The array of RGB colors corresponding to labels

- **labels** :: `string[]`

  The array of names corresponding to labels

- **segmentationMap** :: `tf.Tensor2D`

  The segmentation map of the image

- **canvas** (optional) :: `HTMLCanvasElement`

  Pass an optional canvas element as `canvas` to draw the output

#### `toSegmentationImage(colormap, labels, segmentationMap, canvas?)` outputs

  A promise resolving to the `SegmentationData` object that contains two attributes:

- **legend** :: `{ [name: string]: [number, number, number] }`

  The legend is a dictionary of objects recognized in the image and their colors.

- **segmentationMap** :: `Uint8ClampedArray`

  The colored segmentation map as `Uint8ClampedArray` which can be [fed](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Pixel_manipulation_with_canvas) into `ImageData` and mapped to a canvas.

#### `toSegmentationImage(colormap, labels, segmentationMap, canvas?)` example

```javascript
const base = 'pascal';
const translateSegmentationMap = async (segmentationMap) => {
  return await toSegmentationImage(
      getColormap(base), getLabels(base), segmentationMap)
}
```

## Contributing to the Demo

Please see the demo [documentation](./demo/README.md).

## Technical Details

This model is based on the TensorFlow [implementation](https://github.com/tensorflow/models/tree/master/research/deeplab) of DeepLab v3. You might want to inspect the [conversion script](./scripts/convert_deeplab.sh), or download original pre-trained weights [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). To convert the weights locally, run the script as follows, replacing `dist` with the target directory:

```bash
./scripts/convert_deeplab.sh --target_dir ./scripts/dist
```

Run the usage helper to learn more about the options:

```bash
./scripts/convert_deeplab.sh -h
```
