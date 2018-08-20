# Speech Command Recognizer

The Speech Command Recognizer is a JavaScript module that enables
recognition of spoken commands comprised of simple isolated English
words from a small vocabulary. The default vocabulary includes the following
words: the ten digits from "zero" to "nine", "up", "down", "left", "right",
"go", "stop", "yes", "no", as well as the to additional categories of
"unknown word" and "background noise".

It uses the web browser's
[WebAudio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API).
It is built on top of [TensorFlow.js](https://js.tensorflow.org) and can
perform inference and transfer learning entirely in the browser, using
WebGL GPU acceleration.

The underlying deep neural network has been trained using the
[TensorFlow Speech Commands Dataset](https://www.tensorflow.org/tutorials/sequences/audio_recognition).

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
then start the streaming recognition by calling its `startStreaming()` method.

```js
import * as SpeechCommands from '@tensorflow-models/speech-commands';

// When calling `create()`, you must provide the type of the audio input.
// The two available options are `BROWSER_FFT` and `SOFT_FFT`.
// - BROWSER_FFT uses the browser's native Fourier transform.
// - SOFT_FFT uses JavaScript implementations of Fourier transform
//   (not implemented yet).
const recognizer = SpeechCommands.create('BROWSER_FFT');

// See the array of words that the recognizer is trained to recognize.
console.log(recognizer.wordLabels());

// `startStreaming()` takes two arguments:
// 1. A callback function that is invoked anytime a word is recognized.
// 2. A configuration object with adjustable fields such a
//    - includeSpectrogram
//    - probabilityThreshold
recognizer.startStreaming(result => {
  // - result.scores contains the probability scores that correspond to
  //   recognizer.wordLabels().
  // - result.spectrogram contains the spectrogram of the recognized word.
}, {
  includeSpectrogram: true,
  probabilityThreshold: 0.75
});

// Stop the recognition in 10 seconds.
setTimeout(() => recognizer.stopStreaming(), 10e3);
```

#### Parameters for online streaming recognition

As the example above shows, you can specify optional parameters when calling
`startStreaming()`. The supported parameters are:

* `includeSpectrogram`: Let the callback function be invoked with the
  spectrogram data included in the argument. Default: `false`.
* `probabilityThreshold`: The callback function will be invoked if and only if
  the maximum probability score of all the words is greater than this threshold.
  Default: `0`.
* `invokeCallbackOnNoiseAndUnknown`: Whether the callback function will be
  invoked if the "word" with the maximum probability score is the "unknown"
  or "background noise" token. Default: `false`.

### Offline recognition

To perform offline recognition, you need to have obtained the spectrogram
of an audio snippet through a certain means, e.g., by loading the data
from a .wav file or synthesizing the spectrogram programmatically.
Assuming you have the spectrogram stored in an Array of numbers or
a Float32Array, you can create a `tf.Tensor` object. Note that the
shape of the Tensor must match the expectation of the recognizer instance.
E.g.,

```js
import * as tf from '@tensorflow/tfjs';
import * as SpeechCommands from '@tensorflow-models/speech-commands';

const recognizer = SpeechCommands.create('BROWSER_FFT');

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

tf.tidy(() => {
  const x = tf.tensor4d(
      mySpectrogramData, [1].concat(recognizer.modelInputShape().slice(1)));
  const output = recognizer.recognize(x);
  // output has the same format as `result` in the online streaming example
  // above: the `scores` field contains the probabilities of the words.
});
```

### Preloading model

By default, a recognizer object will load the underlying
tf.Model via HTTP requests to a centralized location, when its
`startStreaming()` or `recognize()` method is called the first time.
You can pre-load the model to reduce the latency of the first calls
to these methods. To do that, use the `ensureModelLoaded()` method of the
recognizer object. The `ensureModelLoaded()` method also "warms up" model after
the model is loaded. "Warm up" means running a few dummy examples through the
model for inference to make sure that the necessary states are set up, so that
subsequent inferences can be fast.

## How to run the demo

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
