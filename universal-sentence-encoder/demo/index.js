/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as use from '@tensorflow-models/universal-sentence-encoder';
import {interpolateReds} from 'd3-scale-chromatic';

const sentences = [
  'I like my phone.', 'Your cellphone looks great.', 'How old are you?',
  'What is your age?', 'An apple a day, keeps the doctors away.',
  'Eating strawberries is healthy.'
];

const init = async () => {
  const model = await use.load();

  document.querySelector('#loading').style.display = 'none';
  renderSentences();

  const embeddings = await model.embed(sentences);

  const matrixSize = 250;
  const cellSize = matrixSize / sentences.length;
  const canvas = document.querySelector('canvas');
  canvas.width = matrixSize;
  canvas.height = matrixSize;

  const ctx = canvas.getContext('2d');

  const xLabelsContainer = document.querySelector('.x-axis');
  const yLabelsContainer = document.querySelector('.y-axis');

  for (let i = 0; i < sentences.length; i++) {
    const labelXDom = document.createElement('div');
    const labelYDom = document.createElement('div');

    labelXDom.textContent = i + 1;
    labelYDom.textContent = i + 1;
    labelXDom.style.left = (i * cellSize + cellSize / 2) + 'px';
    labelYDom.style.top = (i * cellSize + cellSize / 2) + 'px';

    xLabelsContainer.appendChild(labelXDom);
    yLabelsContainer.appendChild(labelYDom);

    for (let j = i; j < sentences.length; j++) {
      const sentenceI = embeddings.slice([i, 0], [1]);
      const sentenceJ = embeddings.slice([j, 0], [1]);
      const sentenceITranspose = false;
      const sentenceJTransepose = true;
      const score =
          sentenceI.matMul(sentenceJ, sentenceITranspose, sentenceJTransepose)
              .dataSync();

      ctx.fillStyle = interpolateReds(score);
      ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
    }
  }
};

init();

const renderSentences = () => {
  sentences.forEach((sentence, i) => {
    const sentenceDom = document.createElement('div');
    sentenceDom.textContent = `${i + 1}) ${sentence}`;
    document.querySelector('#sentences-container').appendChild(sentenceDom);
  });
};
