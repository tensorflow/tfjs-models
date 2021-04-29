# TFJS Task API

_WORK IN PROGRESS_

TFJS task API provides an unified experience for running task-specific models
on the web. It covers popular machine learning tasks, such as sentiment
detection, image classification, pose detection, etc. For each task, you can
choose from various models supported by different runtime systems, such as
TFJS, TFLite, MediaPipe, etc. The model interfaces are specifically designed
for each task. They are intuitive and easy-to-use, even for developers without
any comprehensive ML knowledge. In addition, the API will automatically load
required packages on the fly. You never need to worry about missing dependencies
again.

The following table summarizes all the supported tasks and their models:

(TODO)

# Usage

## Import the package

This package is all you need. The packages required by different models will be
loaded on demand automatically.

### Via NPM

```js
// Import @tensorflow-models/tasks.
import * as tftask from '@tensorflow-models/tasks';
```

### Via a script tag

```html
<!-- Import @tensorflow-models/tasks -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/tasks"></script>
```

## Load model and run inference

The code snippet below shows how to load various models for the
`Image Classification` task:

```js
import * as tftask from '@tensorflow-models/tasks';

// Load the TFJS mobilenet model.
const model1 = await tftask.ImageClassification.MobileNet.TFJS.load({
  backend: 'wasm'});

// Load the TFLite mobilenet model.
const model2 = await tftask.ImageClassification.MobileNet.TFLite.load();

// Load a custom image classification TFLite model.
const model3 = await tftask.ImageClassification.CustomModel.TFLite.load({
  model: 'url/to/your/bird_classifier.tflite'});
```

Since all these models are for the `Image Classification` task, they will have
the same task model type: `ImageClassifier` in this case. Each task model
defines an unique and easy-to-use inference method that fits its task the best.
For example, the `ImageClassiier` task model defines a `classify` method that
takes an image-like element and returns the predicted classes:

```js
const result = model1.classify(document.querySelector(img)!);
console.log(result.classes);
```

# TFLite custom model compatibility

TFLite is supported by the [`@tensorflow/tfjs-tflite`][tfjs-tflite] package that
is built on top of the [TFLite Task Library][tflite task library] and
WebAssembly. As a result, all TFLite custom models should comply with the
metadata requirements of the corresonding task in the TFLite task library.
Check out the "model compatibility requirements" section of the official task
library page. For example, the requirements of `ImageClassifier` can be found
[here][req].

# Performance

For TFJS models, the choice of backend affects the performance the most.
For most cases, the WebGL backend (default) is usually the fastest.

For TFLite models, we use WebAssembly under the hood. It uses [XNNPACK][xnnpack]
to accelerate model inference. To achieve the best performance, use a browser
that supports "WebAssembly SIMD" and "WebAssembly threads". In Chrome, these can
be enabled in `chrome://flags/`. The task API will automatically choose the best
WASM module to load and set the number of threads for best performance based on
the current browser environment.

As of March 2021, XNNPACK works best for non-quantized TFLite models. Quantized
models can still be used, but XNNPACK only supports ADD, CONV_2D,
DEPTHWISE_CONV_2D, and FULLY_CONNECTED ops for models with quantization-aware
training using [TF MOT][mot].

# Development

## Building

```sh
$ yarn
$ yarn build
```

## Testing

```sh
$ yarn test
```

## Deployment
```sh
$ yarn build-npm
# (TODO): publish
```

[req]: https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier#model_compatibility_requirements
[tfjs-tflite]: https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite
[tflite task library]: https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview
[xnnpack]: https://github.com/google/XNNPACK
[mot]: https://www.tensorflow.org/model_optimization/api_docs/python/tfmot
