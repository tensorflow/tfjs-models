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

import * as tf from '@tensorflow/tfjs';
import * as SpeechCommands from '../src/index';

console.log(SpeechCommands);  // DEBUG

let recognizer;

const createRecognizerButton = document.getElementById('create-recognizer');
const startButton = document.getElementById('start');
const statusDisplay = document.getElementById('status-display');

createRecognizerButton.addEventListener('click', () => {
  createRecognizerButton.disabled = true;
  logToStatusDisplay('Creating recognizer...');
  recognizer = SpeechCommands.create('BROWSER_FFT');
  recognizer.ensureModelLoaded()
      .then(() => {
        logToStatusDisplay('Model loaded.');
        const wordLabels = recognizer.wordLabels();
        logToStatusDisplay(`${wordLabels.length} word labels: ${wordLabels}`);
        startButton.disabled = false;

        const params = recognizer.params();
        logToStatusDisplay(`sampleRateHz: ${params.sampleRateHz}`);
        logToStatusDisplay(`fftSize: ${params.fftSize}`);
        logToStatusDisplay(
            `spectrogramDurationMillis: ` +
            `${params.spectrogramDurationMillis.toFixed(2)}`);
      })
      .catch(err => {
        logToStatusDisplay(
            'Failed to load model for recognizer: ' + err.message);
      });
});

startButton.addEventListener('click', () => {
  console.log('Calling startStreaming()');  // DEBUG
  recognizer.startStreaming(result => {

  }).then(() => {
    logToStatusDisplay('Streaming recognition started.');
  }).catch(err => {
    logToStatusDisplay('Failed to start streaming display: ' + err.message);
  });
});

function logToStatusDisplay(message) {
  const date = new Date();
  statusDisplay.value += `[${date.toISOString()}] ` + message + '\n';
  statusDisplay.scrollTop = statusDisplay.scrollHeight;
}
