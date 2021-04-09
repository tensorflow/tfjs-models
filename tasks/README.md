# TFJS task API

_WORK IN PROGRESS_

TFJS task API provides an unified experience for running task-specific TFJS and
TFLite models on the web. Each task has various models for users to choose.
Models within a task will have the same easy-to-use API interface.

The following table summarizes all the supported tasks and their models:

(TODO)

TFLite is supported by the [`@tensorflow/tfjs-tflite`][tfjs-tflite] package that
is built on top of the [TFLite Task Library][tflite task library] and
WebAssembly. As a result, all TFLite custom models should comply with the
metadata requirements of the corresonding task in the TFLite task library.

# Usage

## Import the packages

Other than this package itself, you will also need to import the corresponding
TFJS model package and a TFJS backend if you are loading a TFJS model through
the API. You don't need to import extra packages for TFLite models.

### Via NPM

```js
// Adds the webgl backend.
import '@tensorflow/tfjs-backend-webgl';
// Adds the TFJS mobilenet model (as an example).
import '@tensorflow-models/mobilenet';
// Import @tensorflow-models/tasks.
import {loadTaskModel, MLTask} from '@tensorflow-models/tasks';
```

### Via a script tag

```html
<!-- Adds the webgl backend -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
<!-- Adds the TFJS mobilenet model (as an example) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
<!-- Import @tensorflow-models/tasks -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/tasks"></script>
```

## Load model and run inference

The code snippet below shows how to load various models for the
`Image Classification` task:

```js
import {loadTaskModel, MLTask} from '@tensorflow-models/tasks';

// Load the TFJS mobilenet model.
const model1 = await loadTaskModel(
    MLTask.ImageClassification.TFJSMobileNet);

// Load the TFLite mobilenet model.
const model2 = await loadTaskModel(
    MLTask.ImageClassification.TFLiteMobileNet);

// Load a custom image classification TFLite model.
const model3 = await loadTaskModel(
    MLTask.ImageClassification.TFLiteCustomModel),
    {modelUrl: 'url/to/your/bird_classifier.tflite'});
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

# Performance

For TFJS models, the choice of backend affects the performance the most.
For most cases, the WebGL backend is usually the fastest.

For TFLite models, we use WebAssembly under the hood. It uses [XNNPACK][xnnpack]
to accelerate model inference. To achieve the best performance, use a browser
that supports "WebAssembly SIMD" and "WebAssembly threads". In Chrome, these can
be enabled in `chrome://flags/`. As of March 2021, XNNPACK can only be enabled
for non-quantized TFLite models. Quantized models can still be used, but not
accelerated. Support for quantized model acceleration is in the works.

Setting the number of threads when calling `loadTaskModel` can also help with
the performance. In most cases, the threads count should be the same as the
number of physical cores, which is half of `navigator.hardwareConcurrency` on
many x86-64 processors.

```js
const model = await loadTaskModel(
    MLTask.ImageClassification.TFLiteMobileNet),
    {numThreads: navigator.hardwareConcurrency / 2});
```

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

[tfjs-tflite]: https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite
[tflite task library]: https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview
[xnnpack]: https://github.com/google/XNNPACK
