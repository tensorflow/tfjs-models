# Semantic Segmentation in the Browser: DeepLab Model

This package contains a standalone implementation of the DeepLab inference pipeline, as well as a [demo](./demo), for running semantic segmentation using TensorFlow.js.

![DeepLab Demo](./docs/deeplab-demo.gif)

## Usage

In the first step of semantic segmentation, an image is fed through a pre-trained model [based](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) on MobileNet-v2. Three types of pre-trained weights are available, trained on [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/), [Cityscapes](https://www.cityscapes-dataset.com) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) datasets.

To get started, pick the model name from `pascal`, `cityscapes` and `ade20k`, and decide whether you want your model quantized to 16 bits. Then, initialise the model as follows:

```typescript
import { SemanticSegmentation } from '@tensorflow-models/deeplab';
const model = 'pascal';   // set to your preferred model, out of `pascal`, `cityscapes` and `ade20k`
const isQuantized = true; // set to your preference
const deeplab = SemanticSegmentation(model, isQuantized);
```

The download of weights begins automatically.

## Contributing a demo

Please see the demo [documentation](./demo/README.md).
