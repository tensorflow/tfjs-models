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
// import * as use from '@tensorflow-models/universal-sentence-encoder';
const BASE_DIR = 'https://s3.amazonaws.com/tfjstoxicity/';
const MODEL_URL = BASE_DIR + 'model.json';

const predict = async () => {
  const model = await tf.loadFrozenModel(MODEL_URL);
  console.log(model.inputs);
  // const input = {
  //   //'dense_shape': tf.tensor1d([3, 7], 'int32'),
  //   'Placeholder_1': tf.tensor2d([[0, 0],
  //   [0, 1],
  //   [0, 2],
  //   [0, 3],
  //   [0, 4],
  //   [0, 5],
  //   [0, 6],
  //   [1, 0],
  //   [1, 1],
  //   [1, 2],
  //   [1, 3],
  //   [2, 0],
  //   [2, 1],
  //   [2, 2],
  //   [2, 3]]).asType('int32'),
  //   'Placeholder': tf.tensor1d([87, 11, 241, 56, 1857, 3305, 17, 19, 31, 58, 6888,
  //     32, 11, 746, 221], 'float32')
  // };
  // console.time('First prediction');
  // let result = await model.executeAsync(input);
  // result.forEach((x, i) => { console.log(model.outputs[i].name); x.print() });
  // console.timeEnd('First prediction');

  // for (let i = 0; i < 10; i ++) {
  //   console.time('new prediction' + i);
  //   result = await model.executeAsync(input);
  //   console.timeEnd('new prediction' + i);
  // }
};

predict();