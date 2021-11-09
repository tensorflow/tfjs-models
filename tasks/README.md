# TFJS Task API

_WORK IN PROGRESS_

TFJS Task API provides an unified experience for running task-specific models
on the Web. It is designed with ease-of-use in mind, aiming to improve usability
for JS developers without ML knowledge. It has the following features:

- **Easy-to-discover models**

  Models from different runtime systems (e.g. [TFJS][tfjs], [TFLite][tflite],
  [MediaPipe][mediapipe], etc) are grouped by popular ML tasks, such as
  sentiment detection, image classification, pose detection, etc.

- **Clean and powerful APIs**

  Different tasks come with different API interfaces that are the most intuitive
  to use for that particular task. Models under the same task share the same
  API, making it easy to explore. Inference can be done within just 3 lines of
  code.

- **Simple installation**

  You only need to import this package (<20K in size) to start using the API
  without needing to worry about other dependencies, such as model packages,
  runtimes, backends, etc. They will be dynamically loaded on demand without
  duplication.

The following table summarizes all the supported tasks and their models:

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Model</th>
      <th>Supported runtimes · Docs · Resources</th>
    </tr>
  </thead>
  <tbody>
    <!-- Image classification -->
    <tr>
      <td rowspan="2">
        <b>Image Classification</b>
        <br>
        Identify images into predefined classes.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/VwPOePq">Demo</a>
      </td>
      <td>Mobilenet</td>
      <td>
        <div>
          <span><code>TFJS  </code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:MobilenetTFJS">API doc</a>
        </div>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:MobilenetTFLite">API doc</a>
        </div>
      </td>
    </tr>
    <tr>
      <td>Custom model</td>
      <td>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:ICCustomModelTFLite">API doc</a>
          <span>·</span>
          <a href="https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier#model_compatibility_requirements">Requirements</a>
          <span>·</span>
          <a href="https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1">Model collection</a>
        </div>
      </td>
    </tr>
    <!-- Object detection -->
    <tr>
      <td rowspan="2">
        <b>Object Detection</b>
        <br>
        Localize and identify multiple objects in a single image.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/PopPPXo">Demo</a>
      </td>
      <td>COCO-SSD</td>
      <td>
        <div>
          <span><code>TFJS  </code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:CocoSsdTFJS">API doc</a>
        </div>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:CocoSsdTFLite">API doc</a>
        </div>
      </td>
    </tr>
    <tr>
      <td>Custom model</td>
      <td>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:ODCustomModelTFLite">API doc</a>
          <span>·</span>
          <a href="https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector#model_compatibility_requirements">Requirements</a>
          <span>·</span>
          <a href="https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1">Model collection</a>
        </div>
      </td>
    </tr>
    <!-- Image Segmentation -->
    <tr>
      <td rowspan="2">
        <b>Image Segmentation</b>
        <br>
        Predict associated class for each pixel of an image.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/yLMYVJw">Demo</a>
      </td>
      <td>Deeplab</td>
      <td>
        <div>
          <span><code>TFJS  </code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:DeeplabTFJS">API doc</a>
        </div>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:DeeplabTFLite">API doc</a>
        </div>
      </td>
    </tr>
    <tr>
      <td>Custom model</td>
      <td>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:ISCustomModelTFLite">API doc</a>
          <span>·</span>
          <a href="https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter#model_compatibility_requirements">Requirements</a>
          <span>·</span>
          <a href="https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1">Model collection</a>
        </div>
      </td>
    </tr>
    <!-- Sentiment Detection -->
    <tr>
      <td rowspan="2">
        <b>Sentiment Detection</b>
        <br>
        Detect pre-defined sentiments in a given paragraph of text.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/xxqVMyK">Demo</a>
      </td>
      <td>Toxicity</td>
      <td>
        <div>
          <span><code>TFJS  </code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:ToxicityTFJS">API doc</a>
        </div>
      </td>
    </tr>
    <tr>
      <td>Movie review</td>
      <td>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:MovieReviewTFLite">API doc</a>
        </div>
      </td>
    </tr>
    <!-- NL Classification -->
    <tr>
      <td>
        <b>NL Classification</b>
        <br>
        Identify texts into predefined classes.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/LYWRjRj">Demo</a>
      </td>
      <td>Custom model</td>
      <td>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:NCCustomModelTFLite">API doc</a>
          <span>·</span>
          <a href="https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier#model_compatibility_requirements">Requirements</a>
        </div>
      </td>
    </tr>
    <!-- Question & Answer -->
    <tr>
      <td>
        <b>Question & Answer</b>
        <br>
        Answer questions based on the content of a given passage.
        <br>
        <a href="https://codepen.io/jinjingforever/pen/poeyYqo">Demo</a>
      </td>
      <td>BertQA</td>
      <td>
        <div>
          <span><code>TFJS  </code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:BertQATFJS">API doc</a>
        </div>
        <div>
          <span><code>TFLite</code></span>
          <span>·</span>
          <a href="https://js.tensorflow.org/api_tasks/latest/#class:BertQATFLite">API doc</a>
        </div>
      </td>
    </tr>
  </tbody>
</table>

(The initial version only supports the web browser environment. NodeJS support is
coming soon)


# Usage

## Import the package

This package is all you need. The packages required by different models will be
loaded on demand automatically.

### Via NPM

```js
// Import @tensorflow-models/tasks.
import * as tfTask from '@tensorflow-models/tasks';
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
import * as tfTask from '@tensorflow-models/tasks';

// Load the TFJS mobilenet model.
const model1 = await tfTask.ImageClassification.MobileNet.TFJS.load({
  backend: 'wasm'});

// Load the TFLite mobilenet model.
const model2 = await tfTask.ImageClassification.MobileNet.TFLite.load();

// Load a custom image classification TFLite model.
const model3 = await tfTask.ImageClassification.CustomModel.TFLite.load({
  model: 'url/to/your/bird_classifier.tflite'});
```

Since all these models are for the `Image Classification` task, they will have
the same task model type: [`ImageClassifier`][image classifier interface] in
this case. Each task model's `predict` inference method has an unique and
easy-to-use API interface. For example, in `ImageClassifier`, the method takes an
image-like element and returns the predicted classes:

```js
const result = model1.predict(document.querySelector(img)!);
console.log(result.classes);
```

## TFLite custom model compatibility

TFLite is supported by the [`@tensorflow/tfjs-tflite`][tfjs-tflite] package that
is built on top of the [TFLite Task Library][tflite task library] and
WebAssembly. As a result, all TFLite custom models should comply with the
metadata requirements of the corresonding task in the TFLite task library.
Check out the "model compatibility requirements" section of the official task
library page. For example, the requirements of `ImageClassifier` can be found
[here][req].

See an example of how to use TFLite custom model in the `Load model and run
inference` section above.

# Advanced Topics

## Performance

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

[tfjs]: https://github.com/tensorflow/tfjs
[tflite]: https://www.tensorflow.org/lite
[mediapipe]: https://github.com/google/mediapipe
[req]: https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier#model_compatibility_requirements
[tfjs-tflite]: https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite
[tflite task library]: https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview
[xnnpack]: https://github.com/google/XNNPACK
[mot]: https://www.tensorflow.org/model_optimization/api_docs/python/tfmot
[image classifier interface]: https://github.com/tensorflow/tfjs-models/blob/master/tasks/src/tasks/image_classification/common.ts
