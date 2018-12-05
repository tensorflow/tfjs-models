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

const inferenceModelNameSpan = document.getElementById('inference-model-name');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const predictionCanvas = document.getElementById('prediction-canvas');

const probaThresholdInput = document.getElementById('proba-threshold');
const epochsInput = document.getElementById('epochs');
const fineTuningEpochsInput = document.getElementById('fine-tuning-epochs');

const datasetIOButton = document.getElementById('dataset-io');
const datasetIOInnerDiv = document.getElementById('dataset-io-inner');
const downloadFilesButton = document.getElementById('download-dataset');
const datasetFileInput = document.getElementById('dataset-file-input');
const uploadFilesButton = document.getElementById('upload-dataset');

const modelIOButton = document.getElementById('model-io');
const transferModelSaveLoadInnerDiv = document.getElementById('transfer-model-save-load-inner');
const loadTransferModelButton = document.getElementById('load-transfer-model');
const saveTransferModelButton = document.getElementById('save-transfer-model');
const savedTransferModelsSelect = document.getElementById('saved-transfer-models');
const saveTransferModelNameInput = document.getElementById('transfer-model-name');
const deleteTransferModelButton = document.getElementById('delete-transfer-model');

const BACKGROUND_NOISE_TAG = SpeechCommands.BACKGROUND_NOISE_TAG;

/**
 * Transfer learning-related UI componenets.
 */
const learnWordsInput = document.getElementById('learn-words');
const durationMultiplierSelect = document.getElementById('duration-multiplier');
const enterLearnWordsButton = document.getElementById('enter-learn-words');
const collectButtonsDiv = document.getElementById('collect-words');
const startTransferLearnButton =
    document.getElementById('start-transfer-learn');

const XFER_MODEL_NAME = 'xfer-model';

// Minimum required number of examples per class for transfer learning.
const MIN_EXAMPLES_PER_CLASS = 8;

let recognizer;
let transferWords;
let transferRecognizer;
let transferDurationMultiplier;

(async function() {
  logToStatusDisplay('Creating recognizer...');
  recognizer = SpeechCommands.create('BROWSER_FFT');

  await populateSavedTransferModelsSelect();

  // Make sure the tf.Model is loaded through HTTP. If this is not
  // called here, the tf.Model will be loaded the first time
  // `listen()` is called.
  recognizer.ensureModelLoaded()
      .then(() => {
        startButton.disabled = false;
        enterLearnWordsButton.disabled = false;
        loadTransferModelButton.disabled = false;
        deleteTransferModelButton.disabled = false;

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
    spectrogram = await transferRecognizer.collectExample(
        word, {durationMultiplier: transferDurationMultiplier});
  }

  const exampleCanvas = document.createElement('canvas');
  exampleCanvas.style['display'] = 'inline-block';
  exampleCanvas.style['vertical-align'] = 'middle';
  exampleCanvas.height = 60;
  exampleCanvas.width = 80;
  exampleCanvas.style['padding'] = '3px';
  if (wordDiv.children.length > 1) {
    wordDiv.removeChild(wordDiv.children[wordDiv.children.length - 1]);
  }
  wordDiv.appendChild(exampleCanvas);

  const modelNumFrames = recognizer.modelInputShape()[1];
  await plotSpectrogram(
      exampleCanvas, spectrogram.data, spectrogram.frameSize,
      spectrogram.frameSize, {
        pixelsPerFrame: exampleCanvas.width / modelNumFrames,
        markMaxIntensityFrame:
            transferDurationMultiplier > 1 && word != BACKGROUND_NOISE_TAG
      });

  const button = wordDiv.children[0];
  const displayWord = word === BACKGROUND_NOISE_TAG ? 'noise' : word;
  const exampleCounts = transferRecognizer.countExamples();
  button.textContent = `${displayWord} (${exampleCounts[word]})`;
}

function updateButtonStateAccordingToTransferRecognizer() {
  const exampleCounts = transferRecognizer.countExamples();
  if (transferWords == null) {
    transferWords = Object.keys(exampleCounts);
  }
  const minCountByClass =
      transferWords.map(word => exampleCounts[word] || 0)
          .reduce((prev, current) => current < prev ? current : prev);

  const requiredMinCountPerClass =
      Math.ceil(MIN_EXAMPLES_PER_CLASS / transferDurationMultiplier);
  if (minCountByClass >= requiredMinCountPerClass) {
    startTransferLearnButton.textContent = 'Start transfer learning';
    startTransferLearnButton.disabled = false;
  } else {
    startTransferLearnButton.textContent =
        `Need at least ${requiredMinCountPerClass} examples per word`;
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

  transferDurationMultiplier = durationMultiplierSelect.value;

  learnWordsInput.disabled = true;
  enterLearnWordsButton.disabled = true;
  transferWords = learnWordsInput.value.trim().split(',').map(w => w.trim());
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
  datasetFileInput.disabled = true;
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
  saveTransferModelButton.disabled = false;
  inferenceModelNameSpan.textContent = transferRecognizer.name;
  startTransferLearnButton.textContent = 'Transfer learning complete.';
  saveTransferModelNameInput.disabled = false;
  saveTransferModelNameInput.value = `transfer-model-${getDateString()}`;
  startButton.disabled = false;
});

downloadFilesButton.addEventListener('click', () => {
  const basename = getDateString();
  const artifacts = transferRecognizer.serializeExamples();

  // Trigger downloading of the data .bin file.
  const anchor = document.createElement('a');
  anchor.download = `${basename}.bin`;
  anchor.href = window.URL.createObjectURL(
      new Blob([artifacts], {type: 'application/octet-stream'}));
  anchor.click();
});

/** Get the base name of the downloaded files based on current dataset. */
function getDateString() {
  const d = new Date();
  const year = `${d.getFullYear()}`;
  let month = `${d.getMonth() + 1}`;
  let day = `${d.getDate()}`;
  if (month.length < 2) {
    month = `0${month}`;
  }
  if (day.length < 2) {
    day = `0${day}`;
  }
  let hour = `${d.getHours()}`;
  if (hour.length < 2) {
    hour = `0${hour}`;
  }
  let minute = `${d.getMinutes()}`;
  if (minute.length < 2) {
    minute = `0${minute}`;
  }
  let second = `${d.getSeconds()}`;
  if (second.length < 2) {
    second = `0${second}`;
  }
  return `${year}-${month}-${day}T${hour}.${minute}.${second}`;
}

uploadFilesButton.addEventListener('click', async () => {
  const files = datasetFileInput.files;
  if (files == null || files.length !== 1) {
    throw new Error('Must select exactly one file.');
  }
  const datasetFileReader = new FileReader();
  datasetFileReader.onload = async event => {
    await loadDatasetInTransferRecognizer(event.target.result);
    durationMultiplierSelect.value = `${transferDurationMultiplier}`;
    durationMultiplierSelect.disabled = true;
    enterLearnWordsButton.disabled = true;
  };
  datasetFileReader.onerror = () =>
      console.error(`Failed to binary data from file '${dataFile.name}'.`);
  datasetFileReader.readAsArrayBuffer(files[0]);
});

async function loadDatasetInTransferRecognizer(serialized) {
  if (transferRecognizer == null) {
    transferRecognizer = recognizer.createTransfer(XFER_MODEL_NAME);
  }
  transferRecognizer.loadExamples(serialized);
  const exampleCounts = transferRecognizer.countExamples();
  const transferWords = [];
  for (const label in exampleCounts) {
    transferWords.push(label);
  }
  transferWords.sort();
  learnWordsInput.value = transferWords.join(',');

  // Update the UI state based on the loaded dataset.
  const wordDivs = createWordDivs(transferWords);
  for (const word of transferWords) {
    const examples = transferRecognizer.getExamples(word);
    for (const example of examples) {
      const spectrogram = example.example.spectrogram;
      if (transferDurationMultiplier == null) {
        const modelNumFrames = transferRecognizer.modelInputShape()[1];
        transferDurationMultiplier = Math.round(
            spectrogram.data.length / spectrogram.frameSize / modelNumFrames);
        console.log(
            `Inferred transferDurationMultiplier from uploaded file: ` +
            `${transferDurationMultiplier}`);
      }
      await addExample(wordDivs[word], word, spectrogram);
    }
  }
  updateButtonStateAccordingToTransferRecognizer();
}

async function populateSavedTransferModelsSelect() {
  const savedModelKeys = await SpeechCommands.listSavedTransferModels();
  while (savedTransferModelsSelect.firstChild) {
    savedTransferModelsSelect.removeChild(
        savedTransferModelsSelect.firstChild);
  }
  if (savedModelKeys.length > 0) {
    for (const key of savedModelKeys) {
      const option = document.createElement('option');
      option.textContent = key;
      option.id = key;
      savedTransferModelsSelect.appendChild(option);
    }
    loadTransferModelButton.disabled = false;
  }
}

saveTransferModelButton.addEventListener('click', async () => {
  await transferRecognizer.save(saveTransferModelNameInput.value.trim());
  await populateSavedTransferModelsSelect();
  saveTransferModelButton.textContent = 'Model saved!';
  saveTransferModelButton.disabled = true;
});

loadTransferModelButton.addEventListener('click', async () => {
  const transferModelName = savedTransferModelsSelect.value;
  await recognizer.ensureModelLoaded();
  transferRecognizer = recognizer.createTransfer(transferModelName);
  await transferRecognizer.load();
  inferenceModelNameSpan.textContent = transferModelName;
  learnWordsInput.value = transferRecognizer.wordLabels().join(',');
  learnWordsInput.disabled = true;
  durationMultiplierSelect.disabled = true;
  enterLearnWordsButton.disabled = true;
  saveTransferModelButton.disabled = true;
  loadTransferModelButton.disabled = true;
  loadTransferModelButton.textContent = 'Model loaded!';
});

modelIOButton.addEventListener('click', () => {
  if (modelIOButton.textContent.endsWith(' >>')) {
    transferModelSaveLoadInnerDiv.style.display = 'inline-block';
    modelIOButton.textContent =
        modelIOButton.textContent.replace(' >>', ' <<');
  } else {
    transferModelSaveLoadInnerDiv.style.display = 'none';
    modelIOButton.textContent =
        modelIOButton.textContent.replace(' <<', ' >>');
  }
});

deleteTransferModelButton.addEventListener('click', async () => {
  const transferModelName = savedTransferModelsSelect.value;
  await recognizer.ensureModelLoaded();
  transferRecognizer = recognizer.createTransfer(transferModelName);
  await transferRecognizer.deleteSaved(transferModelName);
  deleteTransferModelButton.disabled = true;
  deleteTransferModelButton.textContent = `Deleted "${transferModelName}"`;
  await populateSavedTransferModelsSelect();
});

datasetIOButton.addEventListener('click', () => {
  if (datasetIOButton.textContent.endsWith(' >>')) {
    datasetIOInnerDiv.style.display = 'inline-block';
    datasetIOButton.textContent =
        datasetIOButton.textContent.replace(' >>', ' <<');
  } else {
    datasetIOInnerDiv.style.display = 'none';
    datasetIOButton.textContent =
        datasetIOButton.textContent.replace(' <<', ' >>');
  }
});

