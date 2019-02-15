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
const BASE_DIR =
    'https://storage.googleapis.com/tfjs-models/savedmodel/toxicity/';
const MODEL_URL = BASE_DIR + 'model.json';

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
  },
  {
    'id': '01ced7301be2a32d',
    'text': 'now join the anti gay hitler rebellion now!'
  }
];

const labels = [
  'toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat',
  'sexual_explicit', 'obscene'
];

const loadVocabulary = async () => {
  const vocabulary = await fetch(
      `https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/vocab.json`);
  return vocabulary.json();
};

let tokenizer, model;

const classify = async (inputs) => {
  const encodings = inputs.map(d => tokenizer.encode(d));

  const indicesArr =
      encodings.map((arr, i) => arr.map((d, index) => [i, index]));

  let flattenedIndicesArr = [];
  for (let i = 0; i < indicesArr.length; i++) {
    flattenedIndicesArr = flattenedIndicesArr.concat(indicesArr[i]);
  }

  const indices = tf.tensor2d(
      flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
  const values = tf.tensor1d(tf.util.flatten(encodings), 'int32');
  let results =
      await model.executeAsync({Placeholder_1: indices, Placeholder: values});

  results = results.map(
      (d, i) => ({name: model.outputs[i].name, data: d.dataSync()}));

  const predictions = inputs.map((d, sampleIndex) => {
    const obj = {'text': d};

    results.forEach((classification, i) => {
      const label = classification.name.split('/')[0];

      if (label) {
        const prediction =
            classification.data.slice(sampleIndex * 2, sampleIndex * 2 + 2);
        obj[label] = prediction[0] > prediction[1] ? false : true;
      }
    });

    return obj;
  });

  return predictions;
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
  const vocabulary = await loadVocabulary();
  model = await tf.loadFrozenModel(MODEL_URL);
  tokenizer = new use.Tokenizer(vocabulary);

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

  document.querySelector('#classify-new-text')
      .addEventListener('click', (e) => {
        const text = document.querySelector('#classify-new-text-input').value;
        const predictions = classify([text]).then(d => {
          addPredictions(d);
        });
      });
};

predict();