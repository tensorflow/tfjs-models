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

import * as speechCommands from '../src';

import {plotSpectrogram} from './ui';

/** Remove the children of a div that do not have the isFixed attribute. */
export function removeNonFixedChildrenFromWordDiv(wordDiv) {
  for (let i = wordDiv.children.length - 1; i >= 0; --i) {
    if (wordDiv.children[i].getAttribute('isFixed') == null) {
      wordDiv.removeChild(wordDiv.children[i]);
    } else {
      break;
    }
  }
}

/**
 * Get the relative x-coordinate of a click event in a canvas.
 *
 * @param {HTMLCanvasElement} canvasElement The canvas in which the click
 *   event happened.
 * @param {Event} event The click event object.
 * @return {number} The relative x-coordinate: a `number` between 0 and 1.
 */
function getCanvasClickRelativeXCoordinate(canvasElement, event) {
  let x;
  if (event.pageX) {
    x = event.pageX;
  } else {
    x = event.clientX + document.body.scrollLeft +
        document.documentElement.scrollLeft;
  }
  x -= canvasElement.offsetLeft;
  return x / canvasElement.width;
}

/**
 * Dataset visualizer that supports
 *
 * - Display of words and spectrograms
 * - Navigation through examples
 * - Deletion of examples
 */
export class DatasetViz {
  /**
   * Constructor of DatasetViz
   *
   * @param {Object} transferRecognizer An instance of
   *   `speechCommands.TransferSpeechCommandRecognizer`.
   * @param {HTMLDivElement} topLevelContainer The div element that
   *   holds the div elements for the individual words. It is assumed
   *   that each element has its "word" attribute set to the word.
   * @param {number} minExamplesPerClass Minimum number of examples
   *   per word class required for the start-transfer-learning button
   *   to be enabled.
   * @param {HTMLButtonElement} startTransferLearnButton The button
   *   which starts the transfer learning when clicked.
   * @param {HTMLBUttonElement} downloadAsFileButton The button
   *   that triggers downloading of the dataset as a file when clicked.
   * @param {number} transferDurationMultiplier Optional duration
   *   multiplier (the ratio between the length of the example
   *   and the length expected by the model.) Defaults to 1.
   */
  constructor(
      transferRecognizer, topLevelContainer, minExamplesPerClass,
      startTransferLearnButton, downloadAsFileButton,
      transferDurationMultiplier = 1) {
    this.transferRecognizer = transferRecognizer;
    this.container = topLevelContainer;
    this.minExamplesPerClass = minExamplesPerClass;
    this.startTransferLearnButton = startTransferLearnButton;
    this.downloadAsFileButton = downloadAsFileButton;
    this.transferDurationMultiplier = transferDurationMultiplier;

    // Navigation indices for the words.
    this.navIndices = {};
  }

  /** Get the set of words in the dataset visualizer. */
  words_() {
    const words = [];
    for (const element of this.container.children) {
      words.push(element.getAttribute('word'));
    }
    return words;
  }

  /**
   * Draw an example.
   *
   * @param {HTMLDivElement} wordDiv The div element for the word. It is assumed
   *   that it contains the word button as the first child and the canvas as the
   *   second.
   * @param {string} word The word of the example being added.
   * @param {SpectrogramData} spectrogram Optional spectrogram data.
   *   If provided, will use it as is. If not provided, will use WebAudio
   *   to collect an example.
   * @param {RawAudio} rawAudio Raw audio waveform. Optional
   * @param {string} uid UID of the example being drawn. Must match the UID
   *   of the example from `this.transferRecognizer`.
   */
  async drawExample(wordDiv, word, spectrogram, rawAudio, uid) {
    if (uid == null) {
      throw new Error('Error: UID is not provided for pre-existing example.');
    }

    removeNonFixedChildrenFromWordDiv(wordDiv);

    // Create the left and right nav buttons.
    const leftButton = document.createElement('button');
    leftButton.textContent = '←';
    wordDiv.appendChild(leftButton);

    const rightButton = document.createElement('button');
    rightButton.textContent = '→';
    wordDiv.appendChild(rightButton);

    // Determine the position of the example in the word of the dataset.
    const exampleUIDs =
        this.transferRecognizer.getExamples(word).map(ex => ex.uid);
    const position = exampleUIDs.indexOf(uid);
    this.navIndices[word] = exampleUIDs.indexOf(uid);

    if (position > 0) {
      leftButton.addEventListener('click', () => {
        this.redraw(word, exampleUIDs[position - 1]);
      });
    } else {
      leftButton.disabled = true;
    }

    if (position < exampleUIDs.length - 1) {
      rightButton.addEventListener('click', () => {
        this.redraw(word, exampleUIDs[position + 1]);
      });
    } else {
      rightButton.disabled = true;
    }

    // Spectrogram canvas.
    const exampleCanvas = document.createElement('canvas');
    exampleCanvas.style['display'] = 'inline-block';
    exampleCanvas.style['vertical-align'] = 'middle';
    exampleCanvas.height = 60;
    exampleCanvas.width = 80;
    exampleCanvas.style['padding'] = '3px';

    // Set up the click callback for the spectrogram canvas. When clicked,
    // the keyFrameIndex will be set.
    if (word !== speechCommands.BACKGROUND_NOISE_TAG) {
      exampleCanvas.addEventListener('click', event => {
        const relativeX =
            getCanvasClickRelativeXCoordinate(exampleCanvas, event);
        const numFrames = spectrogram.data.length / spectrogram.frameSize;
        const keyFrameIndex = Math.floor(numFrames * relativeX);
        console.log(
            `relativeX=${relativeX}; ` +
            `changed keyFrameIndex to ${keyFrameIndex}`);
        this.transferRecognizer.setExampleKeyFrameIndex(uid, keyFrameIndex);
        this.redraw(word, uid);
      });
    }

    wordDiv.appendChild(exampleCanvas);

    const modelNumFrames = this.transferRecognizer.modelInputShape()[1];
    await plotSpectrogram(
        exampleCanvas, spectrogram.data, spectrogram.frameSize,
        spectrogram.frameSize, {
          pixelsPerFrame: exampleCanvas.width / modelNumFrames,
          maxPixelWidth: Math.round(0.4 * window.innerWidth),
          markKeyFrame: this.transferDurationMultiplier > 1 &&
              word !== speechCommands.BACKGROUND_NOISE_TAG,
          keyFrameIndex: spectrogram.keyFrameIndex
        });

    if (rawAudio != null) {
      const playButton = document.createElement('button');
      playButton.textContent = '▶️';
      playButton.addEventListener('click', () => {
        playButton.disabled = true;
        speechCommands.utils.playRawAudio(
            rawAudio, () => playButton.disabled = false);
      });
      wordDiv.appendChild(playButton);
    }

    // Create Delete button.
    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'X';
    wordDiv.appendChild(deleteButton);

    // Callback for delete button.
    deleteButton.addEventListener('click', () => {
      this.transferRecognizer.removeExample(uid);
      // TODO(cais): Smarter logic for which example to draw after deletion.
      // Right now it always redraws the last available one.
      this.redraw(word);
    });

    this.updateButtons_();
  }

  /**
   * Redraw the spectrogram and buttons for a word.
   *
   * @param {string} word The word being redrawn. This must belong to the
   *   vocabulary currently held by the transferRecognizer.
   * @param {string} uid Optional UID for the example to render. If not
   *   specified, the last available example of the dataset will be drawn.
   */
  async redraw(word, uid) {
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
        {} :
        this.transferRecognizer.countExamples();

    if (word in exampleCounts) {
      const examples = this.transferRecognizer.getExamples(word);
      let example;
      if (uid == null) {
        // Example UID is not specified. Draw the last one available.
        example = examples[examples.length - 1];
      } else {
        // Example UID is specified. Find the example and update navigation
        // indices.
        for (let index = 0; index < examples.length; ++index) {
          if (examples[index].uid === uid) {
            example = examples[index];
          }
        }
      }

      const spectrogram = example.example.spectrogram;
      await this.drawExample(
          wordDiv, word, spectrogram, example.example.rawAudio, example.uid);
    } else {
      removeNonFixedChildrenFromWordDiv(wordDiv);
    }

    this.updateButtons_();
  }

  /**
   * Redraw the spectrograms and buttons for all words.
   *
   * For each word, the last available example is rendered.
   **/
  redrawAll() {
    for (const word of this.words_()) {
      this.redraw(word);
    }
  }

  /** Update the button states according to the state of transferRecognizer. */
  updateButtons_() {
    const exampleCounts = this.transferRecognizer.isDatasetEmpty() ?
        {} :
        this.transferRecognizer.countExamples();
    const minCountByClass =
        this.words_()
            .map(word => exampleCounts[word] || 0)
            .reduce((prev, current) => current < prev ? current : prev);

    for (const element of this.container.children) {
      const word = element.getAttribute('word');
      const button = element.children[0];
      const displayWord =
          word === speechCommands.BACKGROUND_NOISE_TAG ? 'noise' : word;
      const exampleCount = exampleCounts[word] || 0;
      if (exampleCount === 0) {
        button.textContent = `${displayWord} (${exampleCount})`;
      } else {
        const pos = this.navIndices[word] + 1;
        button.textContent = `${displayWord} (${pos}/${exampleCount})`;
      }
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

    this.downloadAsFileButton.disabled =
        this.transferRecognizer.isDatasetEmpty();
  }
}
