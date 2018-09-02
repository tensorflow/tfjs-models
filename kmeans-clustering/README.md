# KNN Classifier

This package provides a utility for creating a classifier using the
[K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
algorithm.

This package is different from the other packages in this repository in that it
doesn't provide a model with weights, but rather a utility for constructing a
KNN model using activations from another model or any other tensors you can
associate with a class.

You can see example code [here](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier/demo).

## Usage example

##### via Script Tag

```html
<html>
  <head>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7"></script>
    <!-- Load MobileNet -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@0.1.1"></script>
    <!-- Load KNN Classifier -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier@0.1.0"></script>
 </head>

  <body>
    <img id='class0' src='/images/class0.jpg '/>
    <img id='class1' src='/images/class1.jpg '/>
    <img id='test' src='/images/test.jpg '/>
  </body>
  <!-- Place your code in the script tag below. You can also use an external .js file -->
  <script>

    const init = async function() {
      // Create the classifier.
      const classifier = knnClassifier.create();

      // Load mobilenet.
      const mobilenetModule = await mobilenet.load();

      // Add MobileNet activations to the model repeatedly for all classes.
      const img0 = tf.fromPixels(document.getElementById('class0'));
      const logits0 = mobilenetModule.infer(img0, 'conv_preds');
      classifier.addExample(logits0, 0);

      const img1 = tf.fromPixels(document.getElementById('class1'));
      const logits1 = mobilenetModule.infer(img1, 'conv_preds');
      classifier.addExample(logits1, 1);

      // Make a prediction.
      const x = tf.fromPixels(document.getElementById('test'));
      const xlogits = mobilenetModule.infer(x, 'conv_preds');
      console.log('Predictions:');
      const result = await classifier.predictClass(xlogits);
      console.log(result);
    }

    init();

  </script>
</html>
```

###### via NPM

```js
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Create the classifier.
const classifier = knnClassifier.create();

// Load mobilenet.
const mobilenet = await mobilenetModule.load();

// Add MobileNet activations to the model repeatedly for all classes.
const img0 = tf.fromPixels(...);
const img0 = tf.fromPixels(document.getElementById('class0'));
classifier.addExample(logits, 0);

const img1 = tf.fromPixels(document.getElementById('class1'));
const logits1 = mobilenet.infer(img1, 'conv_preds');
classifier.addExample(logits, 1);

// Make a prediction.
const x = tf.fromPixels(document.getElementById('test'));
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
- **example:** An example to add to the dataset, usually an activation from
  another model.
- **classIndex:** The class index of the example.

#### Making a prediction

```ts
classifier.predictClass(
  input: tf.Tensor,
  k = 3
): Promise<{classIndex: number, confidences: {[classId: number]: number}}>;
```

Args:
- **input:** An example to make a prediction on, usually an activation from
  another model.
- **k:** The K value to use in K-nearest neighbors. The algorithm will first
  find the K nearest examples from those it was previously shown, and then choose
  the class that appears the most as the final prediction for the input example.
  Defaults to 3. If examples < k, k = examples.

Returns an object with a top classIndex, and confidences mapping all class
indices to their confidence.

#### Misc

##### Clear all examples for a class.

```ts
classifier.clearClass(classIndex: number)
```

Args:
- **classIndex:** The class to clear all examples for.

##### Clear all examples from all classes

```ts
classifier.clearAllClasses()
```

##### Get the example count for each class

```ts
classifier.getClassExampleCount(): {[classId: number]: number}
```

Returns an object that maps classId to example count for that class.

##### Get the full dataset, useful for saving state.

```ts
classifier.getClassifierDataset(): {[classId: number]: Tensor2D}
```

##### Set the full dataset, useful for restoring state.

```ts
classifier.setClassifierDataset(dataset: {[classId: number]: Tensor2D})
```

Args:
- **dataset:** The class dataset matrices map. Can be retrieved from
  getClassDatsetMatrices. Useful for restoring state.

##### Get the total number of classes

```ts
classifier.getNumClasses(): number
```

##### Dispose the classifier and all internal state

Clears up WebGL memory. Useful if you no longer need the classifier in your
application.

```ts
classifier.dispose()
```
