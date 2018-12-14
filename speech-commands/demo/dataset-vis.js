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
    if (uid == null) {
      throw new Error('Error: UID is not provided for pre-existing example.');
    }

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
    deleteButton.addEventListener('click', () => {
      this.transferRecognizer.removeExample(uid);
      this.redraw(word);
    });

    this.updateButtons_();
  }

  removeDisplayedExample_(wordDiv) {
    // Preserve the first element, which is the button.
    while (wordDiv.children.length > 1) {
      wordDiv.removeChild(wordDiv.children[wordDiv.children.length - 1]);
    }
  }

  async redraw(word) {
    if (word == null) {
      throw new Error('word is not specified');
    }
    let divIndex;
    for (divIndex = 0; divIndex < this.container.children.length; ++divIndex) {
      if (this.container.children[divIndex].getAttribute('word') === word) {
        break;
      }
    }
    if (divIndex === this.container.children.length) {
      throw new Error(`Cannot find div corresponding to word ${word}`);
    }
    const wordDiv = this.container.children[divIndex];
    const exampleCounts = this.transferRecognizer.isDatasetEmpty() ?
        {} : this.transferRecognizer.countExamples();

    if (word in exampleCounts) {
      const examples = this.transferRecognizer.getExamples(word);
      const example = examples[examples.length - 1];
      // TODO(cais): Logic for which example to draw, depending on navigation
      // history.
      const spectrogram = example.example.spectrogram;
      await this.drawExample(wordDiv, word, spectrogram, example.uid);
    } else {
      this.removeDisplayedExample_(wordDiv);
    }

    this.updateButtons_();
  }

  redrawAll() {
    for (const word of this.words_()) {
      this.redraw(word);
    }
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
