# Speech Command Recognizer

The Speech Command Recognizer is a JavaScript module that enables
recognition of spoken commands comprised of simple isolated English
words from a small vocabulary. The default vocabulary includes the following
words: the ten digits from "zero" to "nine", "up", "down", "left", "right",
"go", "stop", "yes", "no", as well as the additional categories of
"unknown word" and "background noise".

It uses the web browser's
[WebAudio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API).
It is built on top of [TensorFlow.js](https://js.tensorflow.org) and can
perform inference and transfer learning entirely in the browser, using
WebGL GPU acceleration.

The underlying deep neural network has been trained using the
[TensorFlow Speech Commands Dataset](https://www.tensorflow.org/tutorials/sequences/audio_recognition).

For more details on the data set, see:

Warden, P. (2018) "Speech commands: A dataset for limited-vocabulary
speech recognition" https://arxiv.org/pdf/1804.03209.pdf

## API Usage

A speech command recognizer can be used in two ways:

1. **Online streaming recognition**, during which the library automatically
   opens an audio input channel using the browser's
   [`getUserMedia`](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
   and
   [WebAudio](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
   APIs (requesting permission from user) and performs real-time recognition on
   the audio input.
2. **Offline recognition**, in which you provide a pre-constructed TensorFlow.js
   [Tensor](https://js.tensorflow.org/api/latest/#tensor) object or a
   `Float32Array` and the recognizer will return the recognition results.

### Online streaming recognition

To use the speech-command recognizer, first create a recognizer instance,
then start the streaming recognition by calling its `listen()` method.

```js
const tf = require('@tensorflow/tfjs');
const speechCommands = require('@tensorflow-models/speech-commands');

// When calling `create()`, you must provide the type of the audio input.
// The two available options are `BROWSER_FFT` and `SOFT_FFT`.
// - BROWSER_FFT uses the browser's native Fourier transform.
// - SOFT_FFT uses JavaScript implementations of Fourier transform
//   (not implemented yet).
const recognizer = speechCommands.create('BROWSER_FFT');

// Make sure that the underlying model and metadata are loaded via HTTPS
// requests.
await recognizer.ensureModelLoaded();

// See the array of words that the recognizer is trained to recognize.
console.log(recognizer.wordLabels());

// `listen()` takes two arguments:
// 1. A callback function that is invoked anytime a word is recognized.
// 2. A configuration object with adjustable fields such a
//    - includeSpectrogram
//    - probabilityThreshold
//    - includeEmbedding
recognizer.listen(result => {
  // - result.scores contains the probability scores that correspond to
  //   recognizer.wordLabels().
  // - result.spectrogram contains the spectrogram of the recognized word.
}, {
  includeSpectrogram: true,
  probabilityThreshold: 0.75
});

// Stop the recognition in 10 seconds.
setTimeout(() => recognizer.stopListening(), 10e3);
```

#### Vocabularies

When calling `speechCommands.create()`, you can specify the vocabulary
the loaded model will be able to recognize. This is specified as the second,
optional argument to `speechCommands.create()`. For example:

```js
const recognizer = speechCommands.create('BROWSER_FFT', 'directional4w');
```

Currently, the supported vocabularies are:
 - '18w' (default): The 20 item vocaulbary, consisting of:
   'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
   'eight', 'nine', 'up', 'down', 'left', 'right', 'go', 'stop',
   'yes', and 'no', in addition to '_background_noise_' and '_unknown_'.
 - 'directional4w': The four directional words: 'up', 'down', 'left', and
   'right', in addition to '_background_noise_' and '_unknown_'.

'18w' is the default vocabulary.

#### Parameters for online streaming recognition

As the example above shows, you can specify optional parameters when calling
`listen()`. The supported parameters are:

* `overlapFactor`: Controls how often the recognizer performs prediction on
  spectrograms. Must be >=0 and <1 (default: 0.5). For example,
  if each spectrogram is 1000 ms long and `overlapFactor` is set to 0.25,
  the prediction will happen every 250 ms.
* `includeSpectrogram`: Let the callback function be invoked with the
  spectrogram data included in the argument. Default: `false`.
* `probabilityThreshold`: The callback function will be invoked if and only if
  the maximum probability score of all the words is greater than this threshold.
  Default: `0`.
* `invokeCallbackOnNoiseAndUnknown`: Whether the callback function will be
  invoked if the "word" with the maximum probability score is the "unknown"
  or "background noise" token. Default: `false`.
* `includeEmbedding`: Whether an internal activation from the underlying model
  will be included in the callback argument, in addition to the probability
  scores. Note: if this field is set as `true`, the value of
  `invokeCallbackOnNoiseAndUnknown` will be overridden to `true` and the
  value of `probabilityThreshold` will be overridden to `0`.

### Offline recognition

To perform offline recognition, you need to have obtained the spectrogram
of an audio snippet through a certain means, e.g., by loading the data
from a .wav file or synthesizing the spectrogram programmatically.
Assuming you have the spectrogram stored in an Array of numbers or
a Float32Array, you can create a `tf.Tensor` object. Note that the
shape of the Tensor must match the expectation of the recognizer instance.
E.g.,

```js
const tf = require('@tensorflow/tfjs');
const speechCommands = require('@tensorflow-models/speech-commands');

const recognizer = speechCommands.create('BROWSER_FFT');

// Inspect the input shape of the recognizer's underlying tf.Model.
console.log(recognizer.modelInputShape());
// You will get something like [null, 43, 232, 1].
// - The first dimension (null) is an undetermined batch dimension.
// - The second dimension (e.g., 43) is the number of audio frames.
// - The third dimension (e.g., 232) is the number of frequency data points in
//   every frame (i.e., column) of the spectrogram
// - The last dimension (e.g., 1) is fixed at 1. This follows the convention of
//   convolutional neural networks in TensorFlow.js and Keras.

// Inspect the sampling frequency and FFT size:
console.log(recognizer.params().sampleRateHz);
console.log(recognizer.params().fftSize);


const x = tf.tensor4d(
    mySpectrogramData, [1].concat(recognizer.modelInputShape().slice(1)));
const output = await recognizer.recognize(x);
// output has the same format as `result` in the online streaming example
// above: the `scores` field contains the probabilities of the words.

tf.dispose([x, output]);
```

Note that you must provide a spectrogram value to the `recognize()` call
in order to perform the offline recognition. If `recognize()` is called
without a first argument, it will perform one-shot online recognition
by collecting a frame of audio via WebAudio.

### Preloading model

By default, a recognizer object will load the underlying
tf.Model via HTTP requests to a centralized location, when its
`listen()` or `recognize()` method is called the first time.
You can pre-load the model to reduce the latency of the first calls
to these methods. To do that, use the `ensureModelLoaded()` method of the
recognizer object. The `ensureModelLoaded()` method also "warms up" model after
the model is loaded. "Warm up" means running a few dummy examples through the
model for inference to make sure that the necessary states are set up, so that
subsequent inferences can be fast.

### Transfer learning

**Transfer learning** is the process of taking a model trained
previously on a dataset (say dataset A) and applying it on a
different dataset (say dataset B).
To achieve transfer learning, the model needs to be slightly modified and
re-trained on dataset B. However, thanks to the training on
the original dataset (A), the training on the new dataset (B) takes much less
time and computational resource, in addition to requiring a much smaller amount of
data than the original training data. The modification process involves removing the
top (output) dense layer of the original model and keeping the "base" of the
model. Due to its previous training, the base can be used as a good feature
extractor for any data similar to the original training data.
The removed dense layer is replaced with a new dense layer configured
specifically for the new dataset.

The speech-command model is a model suitable for transfer learning on
previously unseen spoken words. The original model has been trained on a relatively
large dataset (~50k examples from 20 classes). It can be used for transfer learning on
words different from the original vocabulary. We provide an API to perform
this type of transfer learning. The steps are listed in the example
code snippet below

```js
const baseRecognizer = speechCommands.create('BROWSER_FFT');
await baseRecognizer.ensureModelLoaded();

// Each instance of speech-command recognizer supports multiple
// transfer-learning models, each of which can be trained for a different
// new vocabulary.
// Therefore we give a name to the transfer-learning model we are about to
// train ('colors' in this case).
const transferRecognizer = baseRecognizer.createTransfer('colors');

// Call `collectExample()` to collect a number of audio examples
// via WebAudio.
await transferRecognizer.collectExample('red');
await transferRecognizer.collectExample('green');
await transferRecognizer.collectExample('blue');
await transferRecognizer.collectExample('red');
// Don't forget to collect some background-noise examples, so that the
// trasnfer-learned model will be able to detect moments of silence.
await transferRecognizer.collectExample('_background_noise_');
await transferRecognizer.collectExample('green');
await transferRecognizer.collectExample('blue');
await transferRecognizer.collectExample('_background_noise_');
// ... You would typically want to put `collectExample`
//     in the callback of a UI button to allow the user to collect
//     any desired number of examples in random order.

// You can check the counts of examples for different words that have been
// collect for this transfer-learning model.
console.log(transferRecognizer.countExamples());
// e.g., {'red': 2, 'green': 2', 'blue': 2, '_background_noise': 2};

// Start training of the transfer-learning model.
// You can specify `epochs` (number of training epochs) and `callback`
// (the Model.fit callback to use during training), among other configuration
// fields.
await transferRecognizer.train({
  epochs: 25,
  callback: {
    onEpochEnd: async (epoch, logs) => {
      console.log(`Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
    }
  }
});

// After the transfer learning completes, you can start online streaming
// recognition using the new model.
await transferRecognizer.listen(result => {
  // - result.scores contains the scores for the new vocabulary, which
  //   can be checked with:
  const words = transferRecognizer.wordLabels();
  // `result.scores` contains the scores for the new words, not the original
  // words.
  for (let i = 0; i < words; ++i) {
    console.log(`score for word '${words[i]}' = ${result.scores[i]}`);
  }
}, {probabilityThreshold: 0.75});

// Stop the recognition in 10 seconds.
setTimeout(() => transferRecognizer.stopListening(), 10e3);
```

### Serialize examples from a transfer recognizer.

Once examples has been collected with a transfer recognizer,
you can export the examples in serialized form with the `serielizedExamples()`
method, e.g.,

```js
const serialized = transferRecognizer.serializeExamples();
```

`serialized` is a binary `ArrayBuffer` amenable to storage and transmission.
It contains the spectrogram data of the examples, as well as metadata such
as word labels.

You can also serialize the examples from a subset of the words in the
transfer recognizer's vocabulary, e.g.,

```js
const serializedWithOnlyFoo = transferRecognizer.serializeExamples('foo');
// Or
const serializedWithOnlyFooAndBar = transferRecognizer.serializeExamples(['foo', 'bar']);
```

The serialized examples can later be loaded into another instance of
transfer recognizer with the `loadExamples()` method, e.g.,

```js
const clearExisting = false;
newTransferRecognizer.loadExamples(serialized, clearExisting);
```

Theo `clearExisting` flag ensures that the examples that `newTransferRecognizer`
already holds are preserved. If `true`, the existing exampels will be cleared.
If `clearExisting` is not specified, it'll default to `false`.

## Live demo

A developer-oriented live demo is available at
[this address](https://storage.googleapis.com/tfjs-speech-model-test/2019-01-03a/dist/index.html).

## How to run the demo from source code

The demo/ folder contains a live demo of the speech-command recognizer.
To run it, do

```sh
cd speech-commands
yarn
yarn publish-local
cd demo
yarn
yarn link-local
yarn watch
```
