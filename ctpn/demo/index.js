/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as mobilenet from '@tensorflow-models/ctpn';

const img = document.getElementById('img');
const canvasPrediction = document.getElementById('prediction');
const config = {
      nms_function: 'TF',
      anchor_scales: [16],
      pixel_means: tf.tensor([[[102.9801, 115.9465, 122.7717]]]),
      scales: [600,] ,
      max_size:  1000,
      has_rpn: true,
      detect_mode: 'O',
      pre_nms_topN: 12000,
      post_nms_topN: 2000,
      nms_thresh:0.7,
      min_size: 8,
};
async function run() {
  // Load the model.
  const model = await ctpn.load();

  // Get the prediction.
  const prediction = await model.predict(img, config);
  canvasPrediction.width = img.width;
  canvasPrediction.height = img.height;
  await model.draw(canvasPrediction, prediction.prediction, prediction.scalefactor, 'red');
  console.log('Prediction');
  prediction.print();
}

run();
