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

// const createRecognizerButton = document.getElementById('create-recognizer');
const XFER_MODEL_NAME = 'xfer-model';

let recognizer;
let transferRecognizer;
let current_activation_word = 'go';
let expected_activation_word = 'stop';

function scroll_down() {
    document.body.scrollTop = document.body.scrollTop + 100;
    document.documentElement.scrollTop = document.documentElement.scrollTop + 100;
}

function scroll_up() {
    document.body.scrollTop = document.body.scrollTop - 100;
    document.documentElement.scrollTop = document.documentElement.scrollTop - 100;
}

function recognize_word_index(word_probabilities) {
    let max_probability;
    let max_probability_index;

    max_probability = word_probabilities[0];
    max_probability_index = 0;

    for(let i = 1; i < word_probabilities.length; i++) {
        if (word_probabilities[i] > max_probability) {
            max_probability = word_probabilities[i];
            max_probability_index = i;
        }
    }

    return max_probability_index;
}

const words = [
  "_background_noise_",
  "_unknown_",
   "down",
    "eight",
    "five",
    "four",
    "go",
    "left",
    "nine",
    "no",
    "one",
    "right",
    "seven",
    "six",
    "stop",
    "three",
    "two",
    "up",
    "yes",
    "zero"];

window.addEventListener('load', async () => {
  // createRecognizerButton.disabled = true;
  console.log('Loading module on page load');
  recognizer = SpeechCommands.create('BROWSER_FFT');

  // Make sure the tf.Model is loaded through HTTP. If this is not
  // called here, the tf.Model will be loaded the first time
  // `startStreaming()` is called.
  recognizer.ensureModelLoaded()
      .then(() => {
          console.log('Model loaded');
      })
      .catch(err => {
          console.log('Fail to load model');
      });

  const activeRecognizer =
        transferRecognizer == null ? recognizer : transferRecognizer;
      // populateCandidateWords(words);

      activeRecognizer
          .startStreaming(
              result => {

                let word_index = recognize_word_index(result.scores);
                console.log('Result object is: ', result);
                console.log('Predicted the word: ', words[word_index]);

                // perform actions based on voice SpeechCommands
                if (words[word_index] === expected_activation_word) {
                    if (current_activation_word === 'go') {
                        current_activation_word = 'stop';
                        expected_activation_word = 'go';
                    } else if (current_activation_word === 'stop') {
                        current_activation_word = 'go';
                        expected_activation_word = 'stop';
                    }
                }

                if (current_activation_word === 'go') {
                    document.getElementById("stop").style.display = "none";
                    document.getElementById("go").style.display = "block";
                    document.getElementById("Text").textContent = "GO!";
                    if (words[word_index] === 'down') {
                        console.log(`Predicted the word 'down', will be scrolling down`);
                        var down = document.getElementById("up");
                        down.style.transform = "rotate(180deg)";
                        down.style.visibility = "visible";
                        console.log("down:");
                        console.log(down);
                        setTimeout(stopUp,2000);
                        scroll_down();
                    } else if (words[word_index] === 'up') {
                        console.log(`Predicted the word 'up', will be scrolling up`);
                        var up = document.getElementById("up");
                        up.style.transform = "";
                        up.style.visibility = "visible";
                        console.log("up:");
                        console.log(up);
                        setTimeout(stopUp,2000);
                        scroll_up();
                    } else {
                        console.log(`Predicted the word ${words[word_index]}, neither scrolling up or down`);
                    }
                }
                else if(current_activation_word === 'stop'){
                    document.getElementById("go").style.display = "none";
                    document.getElementById("stop").style.display = "block";
                    document.getElementById("Text").textContent = "STOP!";
                }
              })
                // plotPredictions(
                //     predictionCanvas, activeRecognizer.wordLabels(), result.scores,
                //     3);
                // plotSpectrogram(
                //     spectrogramCanvas, result.spectrogram.data,
                //     result.spectrogram.frameSize, result.spectrogram.frameSize);
              // },
              // {
              //   includeSpectrogram: true,
              //   probabilityThreshold: Number.parseFloat(probaThresholdInput.value)
              // })
          .then(() => {
            // startButton.disabled = true;
            // stopButton.disabled = false;
            console.log('Streaming recognition started');
            // showCandidateWords();
            // logToStatusDisplay('Streaming recognition started.');
          })
          .catch(err => {
          //   logToStatusDisplay(
          //       'ERROR: Failed to start streaming display: ' + err.message);
          // });
          console.log('Failed to start streaming display');
        });
});


function stopUp(){
    var up = document.getElementById("up");
    up.style.visibility="hidden";
}

// createRecognizerButton.addEventListener('click', async () => {
//   createRecognizerButton.disabled = true;
//   logToStatusDisplay('Creating recognizer...');
//   recognizer = SpeechCommands.create('BROWSER_FFT');
//
//   // Make sure the tf.Model is loaded through HTTP. If this is not
//   // called here, the tf.Model will be loaded the first time
//   // `startStreaming()` is called.
//   recognizer.ensureModelLoaded()
//       .then(() => {
//         startButton.disabled = false;
//         enterLearnWordsButton.disabled = false;
//
//         logToStatusDisplay('Model loaded.');
//
//         const params = recognizer.params();
//         logToStatusDisplay(`sampleRateHz: ${params.sampleRateHz}`);
//         logToStatusDisplay(`fftSize: ${params.fftSize}`);
//         logToStatusDisplay(
//             `spectrogramDurationMillis: ` +
//             `${params.spectrogramDurationMillis.toFixed(2)}`);
//         logToStatusDisplay(
//             `tf.Model input shape: ` +
//             `${JSON.stringify(recognizer.modelInputShape())}`);
//       })
//       .catch(err => {
//         logToStatusDisplay(
//             'Failed to load model for recognizer: ' + err.message);
//       });
// });
//
// startButton.addEventListener('click', () => {
//   const activeRecognizer =
//       transferRecognizer == null ? recognizer : transferRecognizer;
//   populateCandidateWords(activeRecognizer.wordLabels());
//
//   activeRecognizer
//       .startStreaming(
//           result => {
//             plotPredictions(
//                 predictionCanvas, activeRecognizer.wordLabels(), result.scores,
//                 3);
//             plotSpectrogram(
//                 spectrogramCanvas, result.spectrogram.data,
//                 result.spectrogram.frameSize, result.spectrogram.frameSize);
//           },
//           {
//             includeSpectrogram: true,
//             probabilityThreshold: Number.parseFloat(probaThresholdInput.value)
//           })
//       .then(() => {
//         startButton.disabled = true;
//         stopButton.disabled = false;
//         showCandidateWords();
//         logToStatusDisplay('Streaming recognition started.');
//       })
//       .catch(err => {
//         logToStatusDisplay(
//             'ERROR: Failed to start streaming display: ' + err.message);
//       });
// });

// stopButton.addEventListener('click', () => {
//   const activeRecognizer =
//       transferRecognizer == null ? recognizer : transferRecognizer;
//   activeRecognizer.stopStreaming()
//       .then(() => {
//         startButton.disabled = false;
//         stopButton.disabled = true;
//         hideCandidateWords();
//         logToStatusDisplay('Streaming recognition stopped.');
//       })
//       .catch(error => {
//         logToStatusDisplay(
//             'ERROR: Failed to stop streaming display: ' + err.message);
//       });
// });
//
// /**
//  * Transfer learning logic.
//  */
//
// function scrollToPageBottom() {
//   const scrollingElement = (document.scrollingElement || document.body);
//   scrollingElement.scrollTop = scrollingElement.scrollHeight;
// }
//
// let collectWordDivs = {};
// let collectWordButtons = {};
//
// enterLearnWordsButton.addEventListener('click', () => {
//   enterLearnWordsButton.disabled = true;
//   const transferWords =
//       learnWordsInput.value.trim().split(',').map(w => w.trim());
//   if (transferWords == null || transferWords.length <= 1) {
//     logToStatusDisplay('ERROR: Invalid list of transfer words.');
//     return;
//   }
//
//   transferRecognizer = recognizer.createTransfer(XFER_MODEL_NAME);
//
//   for (const word of transferWords) {
//     const wordDiv = document.createElement('div');
//     const button = document.createElement('button');
//     button.style['display'] = 'inline-block';
//     button.style['vertical-align'] = 'middle';
//     button.textContent = `Collect "${word}" sample (0)`;
//     wordDiv.appendChild(button);
//     wordDiv.style['height'] = '100px';
//     collectButtonsDiv.appendChild(wordDiv);
//     collectWordDivs[word] = wordDiv;
//     collectWordButtons[word] = button;
//
//     button.addEventListener('click', async () => {
//       disableAllCollectWordButtons();
//       const spectrogram = await transferRecognizer.collectExample(word);
//       const exampleCanvas = document.createElement('canvas');
//       exampleCanvas.style['display'] = 'inline-block';
//       exampleCanvas.style['vertical-align'] = 'middle';
//       exampleCanvas.style['height'] = '60px';
//       exampleCanvas.style['width'] = '80px';
//       exampleCanvas.style['padding'] = '3px';
//       wordDiv.appendChild(exampleCanvas);
//       plotSpectrogram(
//           exampleCanvas, spectrogram.data, spectrogram.frameSize,
//           spectrogram.frameSize);
//       const exampleCounts = transferRecognizer.countExamples();
//       button.textContent = `Collect "${word}" sample (${exampleCounts[word]})`;
//       logToStatusDisplay(`Collect one sample of word "${word}"`);
//       enableAllCollectWordButtons();
//       if (Object.keys(exampleCounts).length > 1) {
//         startTransferLearnButton.disabled = false;
//       }
//     });
//   }
//   scrollToPageBottom();
// });
//
// function disableAllCollectWordButtons() {
//   for (const word in collectWordButtons) {
//     collectWordButtons[word].disabled = true;
//   }
// }
//
// function enableAllCollectWordButtons() {
//   for (const word in collectWordButtons) {
//     collectWordButtons[word].disabled = false;
//   }
// }
//
// startTransferLearnButton.addEventListener('click', async () => {
//   startTransferLearnButton.disabled = true;
//   startButton.disabled = true;
//
//   const epochs = Number.parseInt(epochsInput.value);
//   const lossValues =
//       {x: [], y: [], name: 'train', mode: 'lines', line: {width: 1}};
//   const accuracyValues =
//       {x: [], y: [], name: 'train', mode: 'lines', line: {width: 1}};
//   function plotLossAndAccuracy(epoch, loss, acc) {
//     lossValues.x.push(epoch);
//     lossValues.y.push(loss);
//     accuracyValues.x.push(epoch);
//     accuracyValues.y.push(acc);
//     Plotly.newPlot('loss-plot', [lossValues], {
//       width: 360,
//       height: 300,
//       xaxis: {title: 'Epoch #'},
//       yaxis: {title: 'Loss'},
//       font: {size: 18}
//     });
//     Plotly.newPlot('accuracy-plot', [accuracyValues], {
//       width: 360,
//       height: 300,
//       xaxis: {title: 'Epoch #'},
//       yaxis: {title: 'Accuracy'},
//       font: {size: 18}
//     });
//     startTransferLearnButton.textContent =
//         `Transfer-learning... (${(epoch / epochs * 1e2).toFixed(0)}%)`;
//     scrollToPageBottom();
//   }
//
//   disableAllCollectWordButtons();
//   await transferRecognizer.train({
//     epochs,
//     callback: {
//       onEpochEnd: async (epoch, logs) => {
//         plotLossAndAccuracy(epoch, logs.loss, logs.acc);
//       }
//     }
//   });
//   startTransferLearnButton.textContent = 'Transfer learning complete.';
//   startButton.disabled = false;
// });
