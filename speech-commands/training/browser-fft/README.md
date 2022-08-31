# Training a TensorFlow.js model for Speech Commands Using Browser FFT

This directory contains two example notebooks. They demonstrate how to train
custom TensorFlow.js audio models and deploy them for inference. The models
trained this way expect inputs to be spectrograms in a format consistent with
[WebAudio's `getFloatFrequencyData`](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/getFloatFrequencyData).
Therefore they can be deployed to the browser using the speech-commands library
for inference.

Specifically,

- [training_custom_audio_model_in_python.ipynb](./training_custom_audio_model_in_python.ipynb)
  contains steps to preprocess a directory with audio examples stored as .wav
  files and the steps in which a tf.keras model can be trained on the
  preprocessed data. It then demonstrates how the trained tf.keras model can be
  converted to a TensorFlow.js `LayersModel` that can be loaded with the
  speech-command library's `create()` API. In addition, the notebook also shows
  the steps to convert the trained tf.keras model to a TFLite model for
  inference on mobile devices.
- [tflite_conversion.ipynb](./tflite_conversion.ipynb) illustrates how
  an audio model trained on [Teachable Machine](https://teachablemachine.withgoogle.com/train/audio)
  can be converted to TFLite directly.
