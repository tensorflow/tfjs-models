# KNN Classifier

This package provides a utility for creating a classifier using the
[K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
algorithm.

This package is different from the other packages in this repository in that it
doesn't provide a model with weights, but rather a utility for constructing a
KNN model using activations from another model or any other tensors you can
associate with a class/label.

You can see example code [here](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier/demo).

## Usage example

##### via Script Tag

```html
<html>
  <head>
    <!-- Load TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <!-- Load MobileNet -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
    <!-- Load KNN Classifier -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier"></script>
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
      const img0 = tf.browser.fromPixels(document.getElementById('class0'));
      const logits0 = mobilenetModule.infer(img0, true);
      classifier.addExample(logits0, 0);

      const img1 = tf.browser.fromPixels(document.getElementById('class1'));
      const logits1 = mobilenetModule.infer(img1, true);
      classifier.addExample(logits1, 1);

      // Make a prediction.
      const x = tf.browser.fromPixels(document.getElementById('test'));
      const xlogits = mobilenetModule.infer(x, true);
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
const tf = require('@tensorflow/tfjs');
const mobilenetModule = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

// Create the classifier.
const classifier = knnClassifier.create();

// Load mobilenet.
const mobilenet = await mobilenetModule.load();

// Add MobileNet activations to the model repeatedly for all classes.
const img0 = tf.browser.fromPixels(document.getElementById('class0'));
const logits0 = mobilenet.infer(img0, true);
classifier.addExample(logits0, 0);

const img1 = tf.browser.fromPixels(document.getElementById('class1'));
const logits1 = mobilenet.infer(img1, true);
classifier.addExample(logits1, 1);

// Make a prediction.
const x = tf.browser.fromPixels(document.getElementById('test'));
const xlogits = mobilenet.infer(x, true);
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
  label: number|string
): void;
```

Args:
- **example:** An example to add to the dataset, usually an activation from
  another model.
- **label:** The label (class name) of the example.

#### Making a prediction

```ts
classifier.predictClass(
  input: tf.Tensor,
  k = 3
): Promise<{label: string, classIndex: number, confidences: {[classId: number]: number}}>;
```

Args:
- **input:** An example to make a prediction on, usually an activation from
  another model.
- **k:** The K value to use in K-nearest neighbors. The algorithm will first
  find the K nearest examples from those it was previously shown, and then choose
  the class that appears the most as the final prediction for the input example.
  Defaults to 3. If examples < k, k = examples.

Returns an object where:
 - `label`: the label (class name) with the most confidence.
 - `classIndex`: the 0-based index of the class (for backwards compatibility).
 - `confidences`: maps each label to their confidence score.

#### Misc

##### Clear all examples for a class.

```ts
classifier.clearClass(label: number|string)
```

Args:
- **label:** The label to clear all examples for.

##### Clear all examples from all classes

```ts
classifier.clearAllClasses()
```

##### Get the example count for each class

```ts
classifier.getClassExampleCount(): {[label: string]: number}
```

Returns an object that maps label name to example count for that label.

##### Get the full dataset, useful for saving state.

```ts
classifier.getClassifierDataset(): {[label: string]: Tensor2D}
```

##### Set the full dataset, useful for restoring state.

```ts
classifier.setClassifierDataset(dataset: {[label: string]: Tensor2D})
```

Args:
- **dataset:** The label dataset matrices map. Can be retrieved from
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
