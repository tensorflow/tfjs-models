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

const probaThresholdInput = document.getElementById('proba-threshold');
const epochsInput = document.getElementById('epochs');
const fineTuningEpochsInput = document.getElementById('fine-tuning-epochs');

const downloadFilesButton = document.getElementById('download-dataset');
const datasetFilesInput = document.getElementById('dataset-files-input');
const uploadFilesButton = document.getElementById('upload-dataset');

const BACKGROUND_NOISE_TAG = '_background_noise_';

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
const MIN_EXAPMLES_PER_CLASS = 8;

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
            suppressionTimeMillis: 1000,
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

 /** Scroll to the bottom of the page */
function scrollToPageBottom() {
  const scrollingElement = (document.scrollingElement || document.body);
  scrollingElement.scrollTop = scrollingElement.scrollHeight;
}

/**
 * Add an example to the UI, record the example via WebAudio if necessary.
 *
 * @param {HTMLDivElement} wordDiv The div element for the word. It is assumed
 *   that it contains the word button as the first child and the canvas as the
 *   second.
 * @param {string} word The word of the example being added.
 * @param {SpectrogramData} spectrogram Optional spectrogram data.
 *   If provided, will use it as is. If not provided, will use WebAudio
 *   to collect an example.
 */
async function addExample(wordDiv, word, spectrogram) {
  if (spectrogram == null) {
    // Collect an example online.
    spectrogram = await transferRecognizer.collectExample(word);
  }

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

  const button = wordDiv.children[0];
  const displayWord = word === BACKGROUND_NOISE_TAG ? 'noise' : word;
  const exampleCounts = transferRecognizer.countExamples();
  button.textContent = `${displayWord} (${exampleCounts[word]})`;
}

function updateButtonStateAccordingToTransferRecognizer() {
  const exampleCounts = transferRecognizer.countExamples();
  const transferWords = Object.keys(exampleCounts);  // DEBUG
  const minCountByClass =
      transferWords.map(word => exampleCounts[word] || 0)
          .reduce((prev, current) => current < prev ? current : prev);

  if (minCountByClass >= MIN_EXAPMLES_PER_CLASS) {
    startTransferLearnButton.textContent = 'Start transfer learning';
    startTransferLearnButton.disabled = false;
  } else {
    startTransferLearnButton.textContent =
        `Need at least ${MIN_EXAPMLES_PER_CLASS} examples per word`;
  }
  downloadFilesButton.disabled = false;
}

let collectWordButtons = {};

/**
 * Create div elements for transfer words.
 *
 * @param {string[]} transferWords The array of transfer words.
 * @returns {Object} An object mapping word to th div element created for it.
 */
function createWordDivs(transferWords) {
  const wordDivs = {};
  for (const word of transferWords) {
    const wordDiv = document.createElement('div');
    wordDivs[word] = wordDiv;
    const button = document.createElement('button');
    button.style['display'] = 'inline-block';
    button.style['vertical-align'] = 'middle';

    const displayWord = word === BACKGROUND_NOISE_TAG ? 'noise' : word;

    button.textContent = `${displayWord} (0)`;
    wordDiv.appendChild(button);
    wordDiv.className = 'transfer-word';
    collectButtonsDiv.appendChild(wordDiv);
    collectWordButtons[word] = button;

    button.addEventListener('click', async () => {
      disableAllCollectWordButtons();
      await addExample(wordDiv, word);
      updateButtonStateAccordingToTransferRecognizer();
      enableAllCollectWordButtons();
    });
  }
  return wordDivs;
}

enterLearnWordsButton.addEventListener('click', () => {
  // We disable the option to upload an existing dataset from files
  // once the "Enter transfer words" button has been clicked.
  // However, the user can still load an existing dataset from
  // files first and keep appending examples to it.
  disableFileUploadControls();

  enterLearnWordsButton.disabled = true;
  const transferWords =
      learnWordsInput.value.trim().split(',').map(w => w.trim());
  if (transferWords == null || transferWords.length <= 1) {
    logToStatusDisplay('ERROR: Invalid list of transfer words.');
    return;
  }

  transferRecognizer = recognizer.createTransfer(XFER_MODEL_NAME);
  createWordDivs(transferWords);

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

function disableFileUploadControls() {
  datasetFilesInput.disabled = true;
  uploadFilesButton.disabled = true;
}

startTransferLearnButton.addEventListener('click', async () => {
  startTransferLearnButton.disabled = true;
  startButton.disabled = true;

  const INITIAL_PHASE = 'initial';
  const FINE_TUNING_PHASE = 'fineTuningPhase';

  const epochs = parseInt(epochsInput.value);
  const fineTuningEpochs = parseInt(fineTuningEpochsInput.value);
  const trainLossValues = {};
  const valLossValues = {};
  const trainAccValues = {};
  const valAccValues = {};

  for (const phase of [INITIAL_PHASE, FINE_TUNING_PHASE]) {
    const phaseSuffix = phase === FINE_TUNING_PHASE ? ' (FT)' : '';
    const lineWidth = phase === FINE_TUNING_PHASE ? 2 : 1;
    trainLossValues[phase] = {
      x: [],
      y: [],
      name: 'train' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    valLossValues[phase] = {
      x: [],
      y: [],
      name: 'val' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    trainAccValues[phase] = {
      x: [],
      y: [],
      name: 'train' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    valAccValues[phase] = {
      x: [],
      y: [],
      name: 'val' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
  }

  function plotLossAndAccuracy(epoch, loss, acc, val_loss, val_acc, phase) {
    const displayEpoch = phase === FINE_TUNING_PHASE ? (epoch + epochs) : epoch;
    trainLossValues[phase].x.push(displayEpoch);
    trainLossValues[phase].y.push(loss);
    trainAccValues[phase].x.push(displayEpoch);
    trainAccValues[phase].y.push(acc);
    valLossValues[phase].x.push(displayEpoch);
    valLossValues[phase].y.push(val_loss);
    valAccValues[phase].x.push(displayEpoch);
    valAccValues[phase].y.push(val_acc);

    Plotly.newPlot(
        'loss-plot',
        [
          trainLossValues[INITIAL_PHASE], valLossValues[INITIAL_PHASE],
          trainLossValues[FINE_TUNING_PHASE], valLossValues[FINE_TUNING_PHASE]
        ],
        {
          width: 480,
          height: 360,
          xaxis: {title: 'Epoch #'},
          yaxis: {title: 'Loss'},
          font: {size: 18}
        });
    Plotly.newPlot(
        'accuracy-plot',
        [
          trainAccValues[INITIAL_PHASE], valAccValues[INITIAL_PHASE],
          trainAccValues[FINE_TUNING_PHASE], valAccValues[FINE_TUNING_PHASE]
        ],
        {
          width: 480,
          height: 360,
          xaxis: {title: 'Epoch #'},
          yaxis: {title: 'Accuracy'},
          font: {size: 18}
        });
    startTransferLearnButton.textContent = phase === INITIAL_PHASE ?
        `Transfer-learning... (${(epoch / epochs * 1e2).toFixed(0)}%)` :
        `Transfer-learning (fine-tuning)... (${
            (epoch / fineTuningEpochs * 1e2).toFixed(0)}%)`

    scrollToPageBottom();
  }

  disableAllCollectWordButtons();
  await transferRecognizer.train({
    epochs,
    validationSplit: 0.25,
    callback: {
      onEpochEnd: async (epoch, logs) => {
        plotLossAndAccuracy(
            epoch, logs.loss, logs.acc, logs.val_loss, logs.val_acc,
            INITIAL_PHASE);
      }
    },
    fineTuningEpochs,
    fineTuningCallback: {
      onEpochEnd: async (epoch, logs) => {
        plotLossAndAccuracy(
            epoch, logs.loss, logs.acc, logs.val_loss, logs.val_acc,
            FINE_TUNING_PHASE);
      }
    }
  });
  startTransferLearnButton.textContent = 'Transfer learning complete.';
  startButton.disabled = false;
});

downloadFilesButton.addEventListener('click', () => {
  const basename = getDatasetFileBasename();
  const artifacts = transferRecognizer.serializeExamples();

  const manifestAnchor = document.createElement('a');
  manifestAnchor.download = `${basename}.json`;
  manifestAnchor.href = window.URL.createObjectURL(
      new Blob([artifacts.manifest], {type: 'application/json'}));
  manifestAnchor.click();

  const weightDataAnchor = document.createElement('a');
  weightDataAnchor.download = `${basename}.bin`;
  weightDataAnchor.href = window.URL.createObjectURL(
      new Blob([artifacts.data], {type: 'application/octet-stream'}));
  weightDataAnchor.click();
});

function getDatasetFileBasename() {
  if (transferRecognizer == null) {
    throw new Error('Transfer model is unset.');
  }
  const items = [];
  const exampleCounts = transferRecognizer.countExamples();
  const labels = Object.keys(exampleCounts);
  labels.sort();
  for (const label of labels) {
    const count = exampleCounts[label];
    items.push(`${count}${label}`);
  }
  return items.join('-');
}

uploadFilesButton.addEventListener('click', async () => {
  const files = datasetFilesInput.files;
  if (files.length !== 2) {
    throw new Error('Must select exactly two files.');
  }

  let manifestFile;
  let dataFile;
  if (files[0].name.endsWith('.json') && files[1].name.endsWith('.bin')) {
    manifestFile = files[0];
    dataFile = files[1];
  } else if (files[0].name.endsWith('.bin') && files[1].name.endsWith('.json')) {
    manifestFile = files[1];
    dataFile = files[0];
  } else {
    throw new Error('Must select a .json file and a .bin file');
  }

  const manifestReader = new FileReader();
  manifestReader.onload = event => {
    const manifest = event.target.result;

    const dataReader = new FileReader();
    dataReader.onload = async event => {
      // tslint:disable-next-line:no-any
      const data = event.target.result;
      await loadDatasetInTransferRecognizer({manifest, data});
    };
    dataReader.onerror = () => console.error(
        `Failed to binary data from file '${dataFile.name}'.`);
    dataReader.readAsArrayBuffer(dataFile);
  };
  manifestReader.onerror = () => console.error(
      `Failed to read manifest JSON from file '${manifestFile.name}'.`);
  manifestReader.readAsText(manifestFile);
});

async function loadDatasetInTransferRecognizer(artifacts) {
  if (transferRecognizer == null) {
    transferRecognizer = recognizer.createTransfer(XFER_MODEL_NAME);
  }
  transferRecognizer.loadExamples(artifacts);
  const exampleCounts = transferRecognizer.countExamples();
  const transferWords = [];
  for (const label in exampleCounts) {
    transferWords.push(label);
  }
  transferWords.sort();
  learnWordsInput.value = transferWords.join(',');

  const wordDivs = createWordDivs(transferWords);
  for (const word of transferWords) {
    const examples = transferRecognizer.getExamples(word);
    for (const example of examples) {
      await addExample(wordDivs[word], word, example.example.spectrogram);
    }
  }
  updateButtonStateAccordingToTransferRecognizer();
}
