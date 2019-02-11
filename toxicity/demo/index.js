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

console.log("lol");
import * as use from '@tensorflow-models/universal-sentence-encoder';
const BASE_DIR = 'https://s3.amazonaws.com/tfjstoxicity/';
const MODEL_URL = BASE_DIR + 'model.json';

const samples = [
  {
    'id': '',
    'text': 'You suck.'
  },
  {
    'id': '',
    'text': 'I thought it was an excellent movie.'
  },
  {
    'id': '',
    'text': 'This ice cream is delicious.'
  }
];

const loadVocabulary = async() => {
  const vocabulary = await fetch(`https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/vocab.json`);
  return vocabulary.json();
}

const predict = async () => {
  const vocabulary = await loadVocabulary();
  const model = await tf.loadFrozenModel(MODEL_URL);

  console.log("hi");
  console.log(use);
  const tokenizer = new use.Tokenizer(vocabulary);
  const encodings = samples.map(d => tokenizer.encode(d.text));
  console.log(encodings);

  const indicesArr =
      encodings.map((arr, i) => arr.map((d, index) => [i, index]));

  let flattenedIndicesArr = [];
  for (let i = 0; i < indicesArr.length; i++) {
    flattenedIndicesArr =
        flattenedIndicesArr.concat(indicesArr[i]);
  }

  const indices = tf.tensor2d(
      flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
  const values = tf.tensor1d(tf.util.flatten(encodings), 'int32');
  const embeddings = await model.executeAsync({
    Placeholder_1: indices,
    Placeholder: values
  });

  console.log("MODEL OUTPUTS");
  /*
  12 - one for each category
   */
  console.log(model.outputs);

  embeddings.forEach((x, i) => {
    console.log('-------');
    console.log(model.outputs[i]);
    console.log(x);
  });

};

predict();