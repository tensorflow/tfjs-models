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

import {BACKGROUND_NOISE_TAG, UNKNOWN_TAG} from '../src';

const statusDisplay = document.getElementById('status-display');
const candidateWordsContainer = document.getElementById('candidate-words');

/**
 * Log a message to a textarea.
 *
 * @param {string} message Message to be logged.
 */
export function logToStatusDisplay(message) {
  const date = new Date();
  statusDisplay.value += `[${date.toISOString()}] ` + message + '\n';
  statusDisplay.scrollTop = statusDisplay.scrollHeight;
}

let candidateWordSpans;

/**
 * Display candidate words in the UI.
 *
 * The background-noise "word" will be omitted.
 *
 * @param {*} words Candidate words.
 */
export function populateCandidateWords(words) {
  candidateWordSpans = {};
  while (candidateWordsContainer.firstChild) {
    candidateWordsContainer.removeChild(candidateWordsContainer.firstChild);
  }

  // const candidatesLabel = document.createElement('span');
  // candidatesLabel.textContent = 'Words to say: ';
  // candidatesLabel.classList.add('candidate-word');
  // candidatesLabel.classList.add('candidate-word-label');
  // candidateWordsContainer.appendChild(candidatesLabel);

  for (const word of words) {
    if (word === BACKGROUND_NOISE_TAG || word === UNKNOWN_TAG) {
      continue;
    }
    const wordSpan = document.createElement('span');
    wordSpan.textContent = word;
    wordSpan.classList.add('candidate-word');
    candidateWordsContainer.appendChild(wordSpan);
    candidateWordSpans[word] = wordSpan;
  }
}

export function showCandidateWords() {
  candidateWordsContainer.classList.remove('candidate-words-hidden');
}

export function hideCandidateWords() {
  candidateWordsContainer.classList.add('candidate-words-hidden');
}

/**
 * Show an audio spectrogram in a canvas.
 *
 * @param {HTMLCanvasElement} canvas The canvas element to draw the
 *   spectrogram in.
 * @param {Float32Array} frequencyData The flat array for the spectrogram
 *   data.
 * @param {number} fftSize Number of frequency points per frame.
 * @param {number} fftDisplaySize Number of frequency points to show. Must be
 *   <= fftSize.
 */
export function plotSpectrogram(
    canvas, frequencyData, fftSize, fftDisplaySize) {
  if (fftDisplaySize == null) {
    fftDisplaySize = fftSize;
  }

  // Get the maximum and minimum.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < frequencyData.length; ++i) {
    const x = frequencyData[i];
    if (x !== -Infinity) {
      if (x < min) {
        min = x;
      }
      if (x > max) {
        max = x;
      }
    }
  }
  if (min >= max) {
    return;
  }

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);

  const numTimeSteps = frequencyData.length / fftSize;
  const pixelWidth = canvas.width / numTimeSteps;
  const pixelHeight = canvas.height / fftDisplaySize;
  for (let i = 0; i < numTimeSteps; ++i) {
    const x = pixelWidth * i;
    const spectrum = frequencyData.subarray(i * fftSize, (i + 1) * fftSize);
    if (spectrum[0] === -Infinity) {
      break;
    }
    for (let j = 0; j < fftDisplaySize; ++j) {
      const y = canvas.height - (j + 1) * pixelHeight;

      let colorValue = (spectrum[j] - min) / (max - min);
      colorValue = Math.pow(colorValue, 3);
      colorValue = Math.round(255 * colorValue);
      const fillStyle =
          `rgb(${colorValue},${255 - colorValue},${255 - colorValue})`;
      context.fillStyle = fillStyle;
      context.fillRect(x, y, pixelWidth, pixelHeight);
    }
  }
}

/**
 * Plot top-K predictions from a speech command recognizer.
 *
 * @param {HTMLCanvasElement} canvas The canvas to render the predictions in.
 * @param {string[]} candidateWords Candidate word array.
 * @param {Float32Array | number[]} probabilities Probability scores from the
 *   speech command recognizer. Must be of the same length as `candidateWords`.
 * @param {number} topK Top _ scores to render.
 */
export function plotPredictions(canvas, candidateWords, probabilities, topK) {
  if (topK != null) {
    let wordsAndProbs = [];
    for (let i = 0; i < candidateWords.length; ++i) {
      wordsAndProbs.push([candidateWords[i], probabilities[i]]);
    }
    wordsAndProbs.sort((a, b) => (b[1] - a[1]));
    wordsAndProbs = wordsAndProbs.slice(0, topK);
    candidateWords = wordsAndProbs.map(item => item[0]);
    probabilities = wordsAndProbs.map(item => item[1]);

    // Highlight the top word.
    const topWord = wordsAndProbs[0][0];
    for (const word in candidateWordSpans) {
      if (word === topWord) {
        candidateWordSpans[word].classList.add('candidate-word-active');
      } else {
        candidateWordSpans[word].classList.remove('candidate-word-active');
      }
    }
  }

  // const context = canvas.getContext('2d');
  // context.clearRect(0, 0, canvas.width, canvas.height);
  // if (probabilities == null) {
  //   return;
  // }

  // const barWidth = canvas.width / candidateWords.length * 0.8;
  // const barGap = canvas.width / candidateWords.length * 0.2;

  // context.font = '24px Arial';
  // context.beginPath();
  // for (let i = 0; i < candidateWords.length; ++i) {
  //   let word = candidateWords[i];
  //   if (word === BACKGROUND_NOISE_TAG) {
  //     word = 'noise';
  //   }
  //   context.fillText(word, i * (barWidth + barGap), 0.95 * canvas.height);
  // }
  // context.stroke();

  // context.beginPath();
  // for (let i = 0; i < probabilities.length; ++i) {
  //   const x = i * (barWidth + barGap);
  //   context.rect(
  //       x, canvas.height * 0.95 * (1 - probabilities[i]), barWidth,
  //       canvas.height * 0.95 * probabilities[i]);
  // }
  // context.stroke();
}
