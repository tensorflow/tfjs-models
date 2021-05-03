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

import 'regenerator-runtime/runtime';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import {ImageClassifier, loadTaskModel, MLTask} from '@tensorflow-models/tasks';

async function start() {
  const img = document.querySelector('img')!;

  // TFJS mobilenet model.
  let tfjsModel: ImageClassifier;
  document.querySelector('.btn.tfjs').addEventListener('click', async () => {
    // Load model.
    const statsEle = document.querySelector('.result-stats.tfjs')!;
    const needWarmup = (tfjsModel === undefined);
    if (!tfjsModel) {
      statsEle.textContent = 'Loading TFJS model...';
      tfjsModel = await loadTaskModel(
          MLTask.ImageClassification.TFJSMobileNet,
          {modelConfig: {version: 2, alpha: 1.0}});
    }

    // Classify.
    if (needWarmup) {
      await tfjsModel.classify(img);
    }
    const start = Date.now();
    const result = await tfjsModel.classify(img);
    statsEle.textContent = `Inference latency: ${Date.now() - start}ms`;
    document.querySelector('.result.tfjs')!.textContent =
        `${result.classes[0].className} (${
            result.classes[0].probability.toFixed(3)})`;
  });

  // TFLite mobilenet model.
  let tfliteModel: ImageClassifier;
  document.querySelector('.btn.tflite').addEventListener('click', async () => {
    // Load model.
    const statsEle = document.querySelector('.result-stats.tflite')!;
    if (!tfliteModel) {
      statsEle.textContent = 'Loading TFLite model...';
      tfliteModel =
          await loadTaskModel(MLTask.ImageClassification.TFLiteMobileNet, {
            version: 2,
            alpha: 1.0,
            numThreads: navigator.hardwareConcurrency / 2,
          });
    }

    // Classify.
    const start = Date.now();
    const result = await tfliteModel.classify(img);
    statsEle.textContent = `Inference latency: ${Date.now() - start}ms`;
    document.querySelector('.result.tflite')!.textContent =
        `${result.classes[0].className} (${
            result.classes[0].probability.toFixed(3)})`;
  });
}

start();
