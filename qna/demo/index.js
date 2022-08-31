/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as qna from '@tensorflow-models/qna';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

let modelPromise = {};
let search;
let input;
let contextDiv;
let answerDiv;

const process = async () => {
  const model = await modelPromise;
  const answers = await model.findAnswers(input.value, contextDiv.value);
  console.log(answers);
  answerDiv.innerHTML =
      answers.map(answer => answer.text + ' (score =' + answer.score + ')')
          .join('<br>');
};

window.onload = () => {
  modelPromise = qna.load();
  input = document.getElementById('question');
  search = document.getElementById('search');
  contextDiv = document.getElementById('context');
  answerDiv = document.getElementById('answer');
  search.onclick = process;

  input.addEventListener('keyup', async (event) => {
    if (event.key === 'Enter') {
      process();
    }
  });
};
