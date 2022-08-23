/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import Plotly from 'plotly.js-dist';

import * as SpeechCommands from '@tensorflow-models/speech-commands';

import {DatasetViz, removeNonFixedChildrenFromWordDiv} from './dataset-vis';
import {hideCandidateWords, logToStatusDisplay, plotPredictions, plotSpectrogram, populateCandidateWords, showCandidateWords} from './ui';

const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const predictionCanvas = document.getElementById('prediction-canvas');

const probaThresholdInput = document.getElementById('proba-threshold');
const epochsInput = document.getElementById('epochs');
const fineTuningEpochsInput = document.getElementById('fine-tuning-epochs');

const datasetIOButton = document.getElementById('dataset-io');
const datasetIOInnerDiv = document.getElementById('dataset-io-inner');
const downloadAsFileButton = document.getElementById('download-dataset');
const datasetFileInput = document.getElementById('dataset-file-input');
const uploadFilesButton = document.getElementById('upload-dataset');

const evalModelOnDatasetButton =
    document.getElementById('eval-model-on-dataset');
const evalResultsSpan = document.getElementById('eval-results');

const modelIOButton = document.getElementById('model-io');
const transferModelSaveLoadInnerDiv =
    document.getElementById('transfer-model-save-load-inner');
const loadTransferModelButton = document.getElementById('load-transfer-model');
const saveTransferModelButton = document.getElementById('save-transfer-model');
const savedTransferModelsSelect =
    document.getElementById('saved-transfer-models');
const deleteTransferModelButton =
    document.getElementById('delete-transfer-model');

const BACKGROUND_NOISE_TAG = SpeechCommands.BACKGROUND_NOISE_TAG;

/**
 * Transfer learning-related UI componenets.
 */
const transferModelNameInput = document.getElementById('transfer-model-name');
const learnWordsInput = document.getElementById('learn-words');
const durationMultiplierSelect = document.getElementById('duration-multiplier');
const enterLearnWordsButton = document.getElementById('enter-learn-words');
const includeTimeDomainWaveformCheckbox =
    document.getElementById('include-audio-waveform');
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

        transferModelNameInput.value = `model-${getDateString()}`;

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

  const suppressionTimeMillis = 1000;
  activeRecognizer
      .listen(
          result => {
            plotPredictions(
                predictionCanvas, activeRecognizer.wordLabels(), result.scores,
                3, suppressionTimeMillis);
          },
          {
            includeSpectrogram: true,
            suppressionTimeMillis,
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

let collectWordButtons = {};
let datasetViz;

function createProgressBarAndIntervalJob(parentElement, durationSec) {
  const progressBar = document.createElement('progress');
  progressBar.value = 0;
  progressBar.style['width'] = `${Math.round(window.innerWidth * 0.25)}px`;
  // Update progress bar in increments.
  const intervalJob = setInterval(() => {
    progressBar.value += 0.05;
  }, durationSec * 1e3 / 20);
  parentElement.appendChild(progressBar);
  return {progressBar, intervalJob};
}

/**
 * Create div elements for transfer words.
 *
 * @param {string[]} transferWords The array of transfer words.
 * @returns {Object} An object mapping word to th div element created for it.
 */
function createWordDivs(transferWords) {
  // Clear collectButtonsDiv first.
  while (collectButtonsDiv.firstChild) {
    collectButtonsDiv.removeChild(collectButtonsDiv.firstChild);
  }
  datasetViz = new DatasetViz(
      transferRecognizer, collectButtonsDiv, MIN_EXAMPLES_PER_CLASS,
      startTransferLearnButton, downloadAsFileButton,
      transferDurationMultiplier);

  const wordDivs = {};
  for (const word of transferWords) {
    const wordDiv = document.createElement('div');
    wordDiv.classList.add('word-div');
    wordDivs[word] = wordDiv;
    wordDiv.setAttribute('word', word);
    const button = document.createElement('button');
    button.setAttribute('isFixed', 'true');
    button.style['display'] = 'inline-block';
    button.style['vertical-align'] = 'middle';

    const displayWord = word === BACKGROUND_NOISE_TAG ? 'noise' : word;

    button.textContent = `${displayWord} (0)`;
    wordDiv.appendChild(button);
    wordDiv.className = 'transfer-word';
    collectButtonsDiv.appendChild(wordDiv);
    collectWordButtons[word] = button;

    let durationInput;
    if (word === BACKGROUND_NOISE_TAG) {
      // Create noise duration input.
      durationInput = document.createElement('input');
      durationInput.setAttribute('isFixed', 'true');
      durationInput.value = '10';
      durationInput.style['width'] = '100px';
      wordDiv.appendChild(durationInput);
      // Create time-unit span for noise duration.
      const timeUnitSpan = document.createElement('span');
      timeUnitSpan.setAttribute('isFixed', 'true');
      timeUnitSpan.classList.add('settings');
      timeUnitSpan.style['vertical-align'] = 'middle';
      timeUnitSpan.textContent = 'seconds';
      wordDiv.appendChild(timeUnitSpan);
    }

    button.addEventListener('click', async () => {
      disableAllCollectWordButtons();
      removeNonFixedChildrenFromWordDiv(wordDiv);

      const collectExampleOptions = {};
      let durationSec;
      let intervalJob;
      let progressBar;

      if (word === BACKGROUND_NOISE_TAG) {
        // If the word type is background noise, display a progress bar during
        // sound collection and do not show an incrementally updating
        // spectrogram.
        // _background_noise_ examples are special, in that user can specify
        // the length of the recording (in seconds).
        collectExampleOptions.durationSec =
            Number.parseFloat(durationInput.value);
        durationSec = collectExampleOptions.durationSec;

        const barAndJob = createProgressBarAndIntervalJob(wordDiv, durationSec);
        progressBar = barAndJob.progressBar;
        intervalJob = barAndJob.intervalJob;
      } else {
        // If this is not a background-noise word type and if the duration
        // multiplier is >1 (> ~1 s recoding), show an incrementally
        // updating spectrogram in real time.
        collectExampleOptions.durationMultiplier = transferDurationMultiplier;
        let tempSpectrogramData;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.style['margin-left'] = '132px';
        tempCanvas.height = 50;
        wordDiv.appendChild(tempCanvas);

        collectExampleOptions.snippetDurationSec = 0.1;
        collectExampleOptions.onSnippet = async (spectrogram) => {
          if (tempSpectrogramData == null) {
            tempSpectrogramData = spectrogram.data;
          } else {
            tempSpectrogramData = SpeechCommands.utils.concatenateFloat32Arrays(
                [tempSpectrogramData, spectrogram.data]);
          }
          plotSpectrogram(
              tempCanvas, tempSpectrogramData, spectrogram.frameSize,
              spectrogram.frameSize, {pixelsPerFrame: 2});
        }
      }

      collectExampleOptions.includeRawAudio =
          includeTimeDomainWaveformCheckbox.checked;
      const spectrogram =
          await transferRecognizer.collectExample(word, collectExampleOptions);


      if (intervalJob != null) {
        clearInterval(intervalJob);
      }
      if (progressBar != null) {
        wordDiv.removeChild(progressBar);
      }
      const examples = transferRecognizer.getExamples(word)
      const example = examples[examples.length - 1];
      await datasetViz.drawExample(
          wordDiv, word, spectrogram, example.example.rawAudio, example.uid);
      enableAllCollectWordButtons();
    });
  }
  return wordDivs;
}

enterLearnWordsButton.addEventListener('click', () => {
  const modelName = transferModelNameInput.value;
  if (modelName == null || modelName.length === 0) {
    enterLearnWordsButton.textContent = 'Need model name!';
    setTimeout(() => {
      enterLearnWordsButton.textContent = 'Enter transfer words';
    }, 2000);
    return;
  }

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
  transferWords.sort();
  if (transferWords == null || transferWords.length <= 1) {
    logToStatusDisplay('ERROR: Invalid list of transfer words.');
    return;
  }

  transferRecognizer = recognizer.createTransfer(modelName);
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
  startTransferLearnButton.textContent = 'Transfer learning starting...';
  await tf.nextFrame();

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
  const augmentByMixingNoiseRatio =
      document.getElementById('augment-by-mixing-noise').checked ? 0.5 : null;
  console.log(`augmentByMixingNoiseRatio = ${augmentByMixingNoiseRatio}`);
  await transferRecognizer.train({
    epochs,
    validationSplit: 0.25,
    augmentByMixingNoiseRatio,
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
  transferModelNameInput.value = transferRecognizer.name;
  transferModelNameInput.disabled = true;
  startTransferLearnButton.textContent = 'Transfer learning complete.';
  transferModelNameInput.disabled = false;
  startButton.disabled = false;
  evalModelOnDatasetButton.disabled = false;
});

downloadAsFileButton.addEventListener('click', () => {
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
    try {
      await loadDatasetInTransferRecognizer(event.target.result);
    } catch (err) {
      const originalTextContent = uploadFilesButton.textContent;
      uploadFilesButton.textContent = err.message;
      setTimeout(() => {
        uploadFilesButton.textContent = originalTextContent;
      }, 2000);
    }
    durationMultiplierSelect.value = `${transferDurationMultiplier}`;
    durationMultiplierSelect.disabled = true;
    enterLearnWordsButton.disabled = true;
  };
  datasetFileReader.onerror = () =>
      console.error(`Failed to binary data from file '${dataFile.name}'.`);
  datasetFileReader.readAsArrayBuffer(files[0]);
});

async function loadDatasetInTransferRecognizer(serialized) {
  const modelName = transferModelNameInput.value;
  if (modelName == null || modelName.length === 0) {
    throw new Error('Need model name!');
  }

  if (transferRecognizer == null) {
    transferRecognizer = recognizer.createTransfer(modelName);
  }
  transferRecognizer.loadExamples(serialized);
  const exampleCounts = transferRecognizer.countExamples();
  transferWords = [];
  const modelNumFrames = transferRecognizer.modelInputShape()[1];
  const durationMultipliers = [];
  for (const word in exampleCounts) {
    transferWords.push(word);
    const examples = transferRecognizer.getExamples(word);
    for (const example of examples) {
      const spectrogram = example.example.spectrogram;
      // Ignore _background_noise_ examples when determining the duration
      // multiplier of the dataset.
      if (word !== BACKGROUND_NOISE_TAG) {
        durationMultipliers.push(Math.round(
            spectrogram.data.length / spectrogram.frameSize / modelNumFrames));
      }
    }
  }
  transferWords.sort();
  learnWordsInput.value = transferWords.join(',');

  // Determine the transferDurationMultiplier value from the dataset.
  transferDurationMultiplier =
      durationMultipliers.length > 0 ? Math.max(...durationMultipliers) : 1;
  console.log(
      `Deteremined transferDurationMultiplier from uploaded ` +
      `dataset: ${transferDurationMultiplier}`);

  createWordDivs(transferWords);
  datasetViz.redrawAll();
}

evalModelOnDatasetButton.addEventListener('click', async () => {
  const files = datasetFileInput.files;
  if (files == null || files.length !== 1) {
    throw new Error('Must select exactly one file.');
  }
  evalModelOnDatasetButton.disabled = true;
  const datasetFileReader = new FileReader();
  datasetFileReader.onload = async event => {
    try {
      if (transferRecognizer == null) {
        throw new Error('There is no model!');
      }

      // Load the dataset and perform evaluation of the transfer
      // model using the dataset.
      transferRecognizer.loadExamples(event.target.result);
      const evalResult = await transferRecognizer.evaluate({
        windowHopRatio: 0.25,
        wordProbThresholds: [
          0,    0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.5,
          0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1.0
        ]
      });
      // Plot the ROC curve.
      const rocDataForPlot = {x: [], y: []};
      evalResult.rocCurve.forEach(item => {
        rocDataForPlot.x.push(item.fpr);
        rocDataForPlot.y.push(item.tpr);
      });

      Plotly.newPlot('roc-plot', [rocDataForPlot], {
        width: 360,
        height: 360,
        mode: 'markers',
        marker: {size: 7},
        xaxis: {title: 'False positive rate (FPR)', range: [0, 1]},
        yaxis: {title: 'True positive rate (TPR)', range: [0, 1]},
        font: {size: 18}
      });
      evalResultsSpan.textContent = `AUC = ${evalResult.auc}`;
    } catch (err) {
      const originalTextContent = evalModelOnDatasetButton.textContent;
      evalModelOnDatasetButton.textContent = err.message;
      setTimeout(() => {
        evalModelOnDatasetButton.textContent = originalTextContent;
      }, 2000);
    }
    evalModelOnDatasetButton.disabled = false;
  };
  datasetFileReader.onerror = () =>
      console.error(`Failed to binary data from file '${dataFile.name}'.`);
  datasetFileReader.readAsArrayBuffer(files[0]);
});

async function populateSavedTransferModelsSelect() {
  const savedModelKeys = await SpeechCommands.listSavedTransferModels();
  while (savedTransferModelsSelect.firstChild) {
    savedTransferModelsSelect.removeChild(savedTransferModelsSelect.firstChild);
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
  await transferRecognizer.save();
  await populateSavedTransferModelsSelect();
  saveTransferModelButton.textContent = 'Model saved!';
  saveTransferModelButton.disabled = true;
});

loadTransferModelButton.addEventListener('click', async () => {
  const transferModelName = savedTransferModelsSelect.value;
  await recognizer.ensureModelLoaded();
  transferRecognizer = recognizer.createTransfer(transferModelName);
  await transferRecognizer.load();
  transferModelNameInput.value = transferModelName;
  transferModelNameInput.disabled = true;
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
    modelIOButton.textContent = modelIOButton.textContent.replace(' >>', ' <<');
  } else {
    transferModelSaveLoadInnerDiv.style.display = 'none';
    modelIOButton.textContent = modelIOButton.textContent.replace(' <<', ' >>');
  }
});

deleteTransferModelButton.addEventListener('click', async () => {
  const transferModelName = savedTransferModelsSelect.value;
  await recognizer.ensureModelLoaded();
  transferRecognizer = recognizer.createTransfer(transferModelName);
  await SpeechCommands.deleteSavedTransferModel(transferModelName);
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
