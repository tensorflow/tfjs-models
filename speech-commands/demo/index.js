/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as SpeechCommands from '../src';

import {logToStatusDisplay, plotPredictions, plotSpectrogram, populateCandidateWords, showCandidateWords, hideCandidateWords} from './ui';

const createRecognizerButton = document.getElementById('create-recognizer');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const predictionCanvas = document.getElementById('prediction-canvas');
const spectrogramCanvas = document.getElementById('spectrogram-canvas');

let recognizer;

createRecognizerButton.addEventListener('click', async () => {
  createRecognizerButton.disabled = true;
  logToStatusDisplay('Creating recognizer...');
  recognizer = SpeechCommands.create('BROWSER_FFT');

  // Make sure the tf.Model is loaded through HTTP. If this is not
  // called here, the tf.Model will be loaded the first time
  // `startStreaming()` is called.
  recognizer.ensureModelLoaded()
      .then(() => {
        startButton.disabled = false;

        logToStatusDisplay('Model loaded.');
        const wordLabels = recognizer.wordLabels();
        logToStatusDisplay(`${wordLabels.length} word labels: ${wordLabels}`);
        populateCandidateWords(wordLabels);

        const params = recognizer.params();
        logToStatusDisplay(`sampleRateHz: ${params.sampleRateHz}`);
        logToStatusDisplay(`fftSize: ${params.fftSize}`);
        logToStatusDisplay(
            `spectrogramDurationMillis: ` +
            `${params.spectrogramDurationMillis.toFixed(2)}`);
        logToStatusDisplay(
            `tf.Model input shape: ` +
            `${JSON.stringify(recognizer.modelInputShape())}`);
      })
      .catch(err => {
        logToStatusDisplay(
            'Failed to load model for recognizer: ' + err.message);
      });
});

startButton.addEventListener('click', () => {
  recognizer
      .startStreaming(
          result => {
            plotPredictions(
                predictionCanvas, recognizer.wordLabels(), result.scores, 3);
            plotSpectrogram(
                spectrogramCanvas, result.spectrogram.data,
                result.spectrogram.frameSize, result.spectrogram.frameSize);
          },
          {includeSpectrogram: true, probabilityThreshold: 0.9})
      .then(() => {
        startButton.disabled = true;
        stopButton.disabled = false;
        showCandidateWords();
        logToStatusDisplay('Streaming recognition started.');
      })
      .catch(err => {
        logToStatusDisplay(
            'ERROR: Failed to start streaming display: ' + err.message);
      });
});

stopButton.addEventListener('click', () => {
  recognizer.stopStreaming()
      .then(() => {
        startButton.disabled = false;
        stopButton.disabled = true;
        hideCandidateWords();
        logToStatusDisplay('Streaming recognition stopped.');
      })
      .catch(error => {
        logToStatusDisplay(
            'ERROR: Failed to stop streaming display: ' + err.message);
      });
});
