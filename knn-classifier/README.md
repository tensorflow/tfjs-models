# KNN Classifier

This package provides a utility for creating a classifier using the
[K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
algorithm.

This package is different from the other packages in this repository in that it
doesn't provide a model with weights, but rather a utility for constructing a
model another models for embeddings.

You can see example code [here](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier/demo).

## Usage

```js
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Create the classifier.
const classifier = knnClassifier.create();

// Load mobilenet.
const mobilenet = await mobilenetModule.load();

// Add embeddings to the model repeatedly for all classes.
const img = tf.fromPixels(...);
const logits = mobilenet.infer(img, 'conv_preds');
const classIndex = 0;
classifier.addExample(logits, 0);

// Make a prediction.
const x = tf.fromPixels(...);
const xlogits = mobilenet.infer(x, 'conv_preds');
console.log('Predictions:');
console.log(classifier.predictClass(xlogits));
```

## API

#### Creating a classifier
`knnClassifier` is the module name, which is automatically included when you use
the <script src> method.

```ts
classifier = knnClassifier.create()
```

Returns a `KNNImageClassifier`.

#### Adding examples

```ts
classifier.addExample(
  example: tf.Tensor,
  classIndex: number
): void;
```

Args:
- **example:** An example to add to the dataset, usually an embedding from
  another model.
- **classIndex:** The class index of the example.

#### Making a prediction

```ts
classifier.predictClass(
  example: tf.Tensor,
  k = 3
): Promise<{classIndex: number, confidences: {[classId: number]: number}}>;
```

Args:
- **example:** An example to make a prediction on, usually an embedding from
  another model.
- **k:** The K value to use in K-nearest neighbors. Determines how many examples
  from the dataset to truncate nearest values by before voting on the best
  class. Defaults to 3.

Returns an object with a top classIndex, and confidences mapping the class index
to the confidence.
