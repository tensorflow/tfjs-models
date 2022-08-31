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

import * as toxicity from '@tensorflow-models/toxicity';

const samples = [
  {
    'id': '002261b0415c4f9d',
    'text':
        'We\'re dudes on computers, moron.  You are quite astonishingly stupid.'
  },
  {
    'id': '0027160ca62626bc',
    'text':
        'Please stop. If you continue to vandalize Wikipedia, as you did to Kmart, you will be blocked from editing.'
  },
  {
    'id': '002fb627b19c4c0b',
    'text':
        'I respect your point of view, and when this discussion originated on 8th April I would have tended to agree with you.'
  }
];

let model, labels;

const classify = async (inputs) => {
  const results = await model.classify(inputs);
  return inputs.map((d, i) => {
    const obj = {'text': d};
    results.forEach((classification) => {
      obj[classification.label] = classification.results[i].match;
    });
    return obj;
  });
};

const addPredictions = (predictions) => {
  const tableWrapper = document.querySelector('#table-wrapper');

  predictions.forEach(d => {
    const predictionDom = `<div class="row">
      <div class="text">${d.text}</div>
      ${
        labels
            .map(
                label => {return `<div class="${
                                 'label' +
                    (d[label] === true ? ' positive' :
                                         '')}">${d[label]}</div>`})
            .join('')}
    </div>`;
    tableWrapper.insertAdjacentHTML('beforeEnd', predictionDom);
  });
};

const predict = async () => {
  model = await toxicity.load();
  labels = model.model.outputNodes.map(d => d.split('/')[0]);

  const tableWrapper = document.querySelector('#table-wrapper');
  tableWrapper.insertAdjacentHTML(
      'beforeend', `<div class="row">
    <div class="text">TEXT</div>
    ${labels.map(label => {
              return `<div class="label">${label.replace('_', ' ')}</div>`;
            }).join('')}
  </div>`);

  const predictions = await classify(samples.map(d => d.text));
  addPredictions(predictions);

  document.querySelector('#classify-new')
      .addEventListener('submit', (e) => {
        const text = document.querySelector('#classify-new-text-input').value;
        const predictions = classify([text]).then(d => {
          addPredictions(d);
        });

        // Prevent submitting the form which would cause a page reload.
        e.preventDefault();
      });
};

predict();
