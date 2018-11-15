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

import Plotly from 'plotly.js-dist';

import * as SpeechCommands from '../src';

import {hideCandidateWords, logToStatusDisplay, plotPredictions, plotSpectrogram, populateCandidateWords, showCandidateWords} from './ui';

const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const predictionCanvas = document.getElementById('prediction-canvas');
const spectrogramCanvas = document.getElementById('spectrogram-canvas');

const probaThresholdInput = document.getElementById('proba-threshold');
const epochsInput = document.getElementById('epochs');

/**
 * Transfer learning-related UI componenets.
 */
const learnWordsInput = document.getElementById('learn-words');
const enterLearnWordsButton = document.getElementById('enter-learn-words');
const collectButtonsDiv = document.getElementById('collect-words');
const startTransferLearnButton =
    document.getElementById('start-transfer-learn');

const XFER_MODEL_NAME = 'xfer-model';

// Minimum required number of examples per class for transfer learning.
// TODO(cais): Change to 8 before submitting. DO NOT SUBMIT.
const MIN_EXAPMLES_PER_CLASS = 4;

let recognizer;
let transferRecognizer;

(async function() {
  logToStatusDisplay('Creating recognizer...');
  recognizer = SpeechCommands.create('BROWSER_FFT');

  // Make sure the tf.Model is loaded through HTTP. If this is not
  // called here, the tf.Model will be loaded the first time
  // `listen()` is called.
  recognizer.ensureModelLoaded()
      .then(() => {
        startButton.disabled = false;
        enterLearnWordsButton.disabled = false;

        logToStatusDisplay('Model loaded.');

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
})();

startButton.addEventListener('click', () => {
  const activeRecognizer =
      transferRecognizer == null ? recognizer : transferRecognizer;
  populateCandidateWords(activeRecognizer.wordLabels());

  activeRecognizer
      .listen(
          result => {
            plotPredictions(
                predictionCanvas, activeRecognizer.wordLabels(), result.scores,
                3);
          },
          {
            includeSpectrogram: true,
            probabilityThreshold: Number.parseFloat(probaThresholdInput.value)
          })
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
  const activeRecognizer =
      transferRecognizer == null ? recognizer : transferRecognizer;
  activeRecognizer.stopListening()
      .then(() => {
        startButton.disabled = false;
        stopButton.disabled = true;
        hideCandidateWords();
        logToStatusDisplay('Streaming recognition stopped.');
      })
      .catch(err => {
        logToStatusDisplay(
            'ERROR: Failed to stop streaming display: ' + err.message);
      });
});

/**
 * Transfer learning logic.
 */

function scrollToPageBottom() {
  const scrollingElement = (document.scrollingElement || document.body);
  scrollingElement.scrollTop = scrollingElement.scrollHeight;
}

let collectWordDivs = {};
let collectWordButtons = {};

enterLearnWordsButton.addEventListener('click', () => {
  enterLearnWordsButton.disabled = true;
  const transferWords =
      learnWordsInput.value.trim().split(',').map(w => w.trim());
  if (transferWords == null || transferWords.length <= 1) {
    logToStatusDisplay('ERROR: Invalid list of transfer words.');
    return;
  }

  transferRecognizer = recognizer.createTransfer(XFER_MODEL_NAME);

  for (const word of transferWords) {
    const wordDiv = document.createElement('div');
    const button = document.createElement('button');
    button.style['display'] = 'inline-block';
    button.style['vertical-align'] = 'middle';

    const displayWord = word === '_background_noise_' ? 'noise' : word;

    button.textContent = `${displayWord} (0)`;
    wordDiv.appendChild(button);
    wordDiv.className = 'transfer-word';
    collectButtonsDiv.appendChild(wordDiv);
    collectWordDivs[word] = wordDiv;
    collectWordButtons[word] = button;

    button.addEventListener('click', async () => {
      disableAllCollectWordButtons();
      const spectrogram = await transferRecognizer.collectExample(word);
      const exampleCanvas = document.createElement('canvas');
      exampleCanvas.style['display'] = 'inline-block';
      exampleCanvas.style['vertical-align'] = 'middle';
      exampleCanvas.style['height'] = '60px';
      exampleCanvas.style['width'] = '80px';
      exampleCanvas.style['padding'] = '3px';
      if (wordDiv.children.length > 1) {
        wordDiv.removeChild(wordDiv.children[wordDiv.children.length - 1]);
      }
      wordDiv.appendChild(exampleCanvas);
      plotSpectrogram(
          exampleCanvas, spectrogram.data, spectrogram.frameSize,
          spectrogram.frameSize);
      const exampleCounts = transferRecognizer.countExamples();
      const classNames = Object.keys(exampleCounts);
      let minCountByClass = Infinity;
      for (const className of classNames) {
        if (exampleCounts[className] < minCountByClass) {
          minCountByClass = exampleCounts[className];
        }
      }
      button.textContent = `${displayWord} (${exampleCounts[word]})`;
      logToStatusDisplay(`Collect one sample of word "${word}"`);
      enableAllCollectWordButtons();
      if (classNames.length > 1 && minCountByClass >= MIN_EXAPMLES_PER_CLASS) {
        startTransferLearnButton.textContent = 'Start transfer learning';
        startTransferLearnButton.disabled = false;
      } else {
        startTransferLearnButton.textContent =
            `Need at least ${MIN_EXAPMLES_PER_CLASS} examples per word`;
      }
    });
  }
  scrollToPageBottom();
});

function disableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = true;
  }
}

function enableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = false;
  }
}

startTransferLearnButton.addEventListener('click', async () => {
  startTransferLearnButton.disabled = true;
  startButton.disabled = true;

  const epochs = Number.parseInt(epochsInput.value);
  const trainLossValues =
      {x: [], y: [], name: 'train', mode: 'lines', line: {width: 1}};
  const valLossValues =
      {x: [], y: [], name: 'val', mode: 'lines', line: {width: 1}};
  const trainAccValues =
      {x: [], y: [], name: 'train', mode: 'lines', line: {width: 1}};
  const valAccValues =
      {x: [], y: [], name: 'val', mode: 'lines', line: {width: 1}};
  function plotLossAndAccuracy(epoch, loss, acc, val_loss, val_acc) {
    trainLossValues.x.push(epoch);
    trainLossValues.y.push(loss);
    trainAccValues.x.push(epoch);
    trainAccValues.y.push(acc);
    valLossValues.x.push(epoch);
    valLossValues.y.push(val_loss);
    valAccValues.x.push(epoch);
    valAccValues.y.push(val_acc);
    Plotly.newPlot('loss-plot', [trainLossValues, valLossValues], {
      width: 360,
      height: 300,
      xaxis: {title: 'Epoch #'},
      yaxis: {title: 'Loss'},
      font: {size: 18}
    });
    Plotly.newPlot('accuracy-plot', [trainAccValues, valAccValues], {
      width: 360,
      height: 300,
      xaxis: {title: 'Epoch #'},
      yaxis: {title: 'Accuracy'},
      font: {size: 18}
    });
    startTransferLearnButton.textContent =
        `Transfer-learning... (${(epoch / epochs * 1e2).toFixed(0)}%)`;
    scrollToPageBottom();
  }

  disableAllCollectWordButtons();
  await transferRecognizer.train({
    epochs,
    validationSplit: 0.25,
    callback: {
      onEpochEnd: async (epoch, logs) => {
        plotLossAndAccuracy(
            epoch, logs.loss, logs.acc, logs.val_loss, logs.val_acc);
        // TODO(cais): Plot fine-tune values.
      }
    },
    fineTuningEpochs: 100,
    fineTuningCallback: {
      onEpochEnd: async (epoch, logs) => {
        // DEBUG
        console.log('Fine-tuning:', epoch, logs.val_loss, logs.val_acc);
      }
    }
  });
  startTransferLearnButton.textContent = 'Transfer learning complete.';
  startButton.disabled = false;
});
