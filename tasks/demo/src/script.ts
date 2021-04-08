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

import {ImageClassifier, loadTaskModel, ML_TASK} from '@tensorflow-models/tasks';

async function start() {
  const img = document.querySelector('img')!;

  // TFJS model.
  let tfjsModel: ImageClassifier;
  document.querySelector('.btn.tfjs').addEventListener('click', async () => {
    // Load model.
    const statsEle = document.querySelector('.result-stats.tfjs')!;
    const needWarmup = (tfjsModel === undefined);
    if (!tfjsModel) {
      statsEle.textContent = 'Loading TFJS model...';
      tfjsModel = await loadTaskModel(
          ML_TASK.ImageClassifier_TFJS,
          {mobilentConfig: {version: 2, alpha: 1.0}});
    }

    // Classify.
    if (needWarmup) {
      await tfjsModel.classify(img);
    }
    const start = Date.now();
    const result = await tfjsModel.classify(img);
    statsEle.textContent = `${Date.now() - start}ms`;
    document.querySelector('.result.tfjs')!.textContent =
        `${result.classes[0].className} (${
            result.classes[0].probability.toFixed(3)})`;
  });

  // TFLite model.
  let tfliteModel: ImageClassifier;
  document.querySelector('.btn.tflite').addEventListener('click', async () => {
    // Load model. This will be fast because it is loading from localhost.
    //
    // TODO: load from tfhub when it is ready.
    const statsEle = document.querySelector('.result-stats.tflite')!;
    if (!tfliteModel) {
      tfliteModel = await loadTaskModel(ML_TASK.ImageClassifier_TFLite, {
        modelPath: 'mobilenetv2.tflite',
        numThreads: navigator.hardwareConcurrency / 2
      });
    }

    // Classify.
    const start = Date.now();
    const result = await tfliteModel.classify(img);
    statsEle.textContent = `${Date.now() - start}ms`;
    document.querySelector('.result.tflite')!.textContent =
        `${result.classes[0].className} (${
            result.classes[0].probability.toFixed(3)})`;
  });
}

start();
