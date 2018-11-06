'use strict';

async function app() {
  console.log('Loading speech commands...')

  // Load the model.
  const recognizer = SpeechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  console.log('Sucessfully loaded model');

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
    // - result.spectrogram contains the spectrogram of the recognized
    // word.
    console.log(result);
  }, {includeSpectrogram: true, probabilityThreshold: 0.75});
}

app();
