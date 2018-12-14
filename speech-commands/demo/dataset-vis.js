import { ConnectableObservable } from "rx";

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

import * as speechCommands from '../src';

import {plotSpectrogram} from './ui';

export class DatasetViz {
  constructor(transferRecognizer,
              topLevelContainer,
              minExamplesPerClass,
              startTransferLearnButton,
              downloadAsFileButton,
              transferDurationMultiplier = 1) {
    console.log('In DataViz ctor()');  // DEBUG
    this.transferRecognizer = transferRecognizer;
    this.container = topLevelContainer;
    this.minExamplesPerClass = minExamplesPerClass;
    this.startTransferLearnButton = startTransferLearnButton;
    this.downloadAsFileButton = downloadAsFileButton;
    this.transferDurationMultiplier = transferDurationMultiplier;
  }

  words_() {
    const words = [];
    for (const element of this.container.children) {
      words.push(element.getAttribute('word'));
    }
    console.log(`words = ${JSON.stringify(words)}`);  // DEBUG
    return words;
  }

  /**
   * UI Component for Visualizing and Manipulating Examples in Dataset.
   */

  /**
   * Draw an example to the UI, record the example via WebAudio if necessary.
   *
   * @param {HTMLDivElement} wordDiv The div element for the word. It is assumed
   *   that it contains the word button as the first child and the canvas as the
   *   second.
   * @param {string} word The word of the example being added.
   * @param {SpectrogramData} spectrogram Optional spectrogram data.
   *   If provided, will use it as is. If not provided, will use WebAudio
   *   to collect an example.
   */
  async drawExample(wordDiv, word, spectrogram, uid) {
    // if (spectrogram == null) {
    //   // Collect an example online.
    //   // TODO(cais): Remove.
    //   // spectrogram = await transferRecognizer.collectExample(
    //   //     word, {durationMultiplier: transferDurationMultiplier});
    //   // const examples = transferRecognizer.getExamples(word)
    //   // exampleUID = examples[examples.length - 1].uid;
    // } else {
    if (uid == null) {
      throw new Error('Error: UID is not provided for pre-existing example.');
    }
    // }

    // Spectrogram canvas.
    const exampleCanvas = document.createElement('canvas');
    exampleCanvas.style['display'] = 'inline-block';
    exampleCanvas.style['vertical-align'] = 'middle';
    exampleCanvas.height = 60;
    exampleCanvas.width = 80;
    exampleCanvas.style['padding'] = '3px';
    this.removeDisplayedExample_(wordDiv);
    wordDiv.appendChild(exampleCanvas);

    const modelNumFrames = this.transferRecognizer.modelInputShape()[1];
    console.log(`modelNumFrames = ${modelNumFrames}`);  // DEBUG
    await plotSpectrogram(
        exampleCanvas, spectrogram.data, spectrogram.frameSize,
        spectrogram.frameSize, {
          pixelsPerFrame: exampleCanvas.width / modelNumFrames,
          markMaxIntensityFrame:
              this.transferDurationMultiplier > 1 &&
                  word !== speechCommands.BACKGROUND_NOISE_TAG
        });

    // Create Delete button.
    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'X';
    wordDiv.appendChild(deleteButton);

    // Callback for delete button.
    console.log('1:', this.transferRecognizer.countExamples());  // DEBUG
    deleteButton.addEventListener('click', () => {
      this.transferRecognizer.removeExample(uid);
      this.redraw(word);  // TODO(cais): Remove.
    });

    // TODO(cais): Confirm removal.
    // const button = wordDiv.children[0];
    // const exampleCounts = this.transferRecognizer.countExamples();
    this.updateButtons_();
  }

  removeDisplayedExample_(wordDiv) {
    // Preserve the first element, which is the button.
    while (wordDiv.children.length > 1) {
      wordDiv.removeChild(wordDiv.children[wordDiv.children.length - 1]);
    }
  }

  async redraw(word) {
    console.log(`In redraw: word = ${word}`);  // DEBUG
    let divIndex;
    for (divIndex = 0; divIndex < this.container.children.length; ++divIndex) {
      if (this.container.children[divIndex].getAttribute('word') === word) {
        break;
      }
    }
    if (divIndex) {
      console.log(`divIndex = ${divIndex}`);  // DEBUG
    }
    // TODO(cais): What is wordDiv is not found?
    const wordDiv = this.container.children[divIndex];
    console.log(`wordDiv =`, wordDiv);  // DEBUG
    // TODO(cais): Deal with this.transferRecognizer.isEmpty();
    const exampleCounts = this.transferRecognizer.isDatasetEmpty() ?
        {} : this.transferRecognizer.countExamples();
    // TODO(cais): What is wordDiv is not found?

    // TODO(cais): Remove div.
    // if (transferRecognizerVocab.indexOf(word) === -1) {
    //   return;
    // }
    if (word in exampleCounts) {
      const examples = this.transferRecognizer.getExamples(word);
      const example = examples[examples.length - 1];
      // TODO(cais): Logic for which example to draw, depending on history.
      const spectrogram = example.example.spectrogram;
      // TODO(cais): This inference logic needs to be redone. DO NOT SUBMIT.
      // if (this.transferDurationMultiplier == null) {
      //   const modelNumFrames = this.transferRecognizer.modelInputShape()[1];
      //   transferDurationMultiplier = Math.round(
      //       spectrogram.data.length / spectrogram.frameSize / modelNumFrames);
      //   console.log(
      //       `Inferred transferDurationMultiplier from uploaded file: ` +
      //       `${transferDurationMultiplier}`);
      // }
      console.log(`Drawing example for word ${word}: ${example.uid}`);  // DEBUG
      await this.drawExample(wordDiv, word, spectrogram, example.uid);
    } else {
      this.removeDisplayedExample_(wordDiv);
    }

    this.updateButtons_();
  }

  updateButtons_() {
    const exampleCounts = this.transferRecognizer.isDatasetEmpty() ?
        {} : this.transferRecognizer.countExamples();
    const minCountByClass =
        this.words_().map(word => exampleCounts[word] || 0)
            .reduce((prev, current) => current < prev ? current : prev);

    for (const element of this.container.children) {
      const word = element.getAttribute('word');
      const button = element.children[0];
      const displayWord = word ===
        speechCommands.BACKGROUND_NOISE_TAG ? 'noise' : word;
      const exampleCount = exampleCounts[word] || 0;
      button.textContent = `${displayWord} (${exampleCount})`;
    }

    const requiredMinCountPerClass =
        Math.ceil(this.minExamplesPerClass / this.transferDurationMultiplier);
    if (minCountByClass >= requiredMinCountPerClass) {
      this.startTransferLearnButton.textContent = 'Start transfer learning';
      this.startTransferLearnButton.disabled = false;
    } else {
      this.startTransferLearnButton.textContent =
          `Need at least ${requiredMinCountPerClass} examples per word`;
      this.startTransferLearnButton.disabled = true;
    }

    this.downloadAsFileButton.disabled = this.transferRecognizer.isDatasetEmpty();
  }
}
