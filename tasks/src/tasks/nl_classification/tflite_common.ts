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

import * as tflite from '@tensorflow/tfjs-tflite';
import {Class} from '../common';
import {NLClassificationResult, NLClassifier} from './common';

/**
 * The base class for all NL classification TFLite models.
 *
 * @template T The type of inference options.
 */
export class NLClassifierTFLite<T> extends NLClassifier<T> {
  constructor(private tfliteNLClassifier: tflite.NLClassifier) {
    super();
  }

  async predict(text: string, infereceOptions?: T):
      Promise<NLClassificationResult> {
    if (!this.tfliteNLClassifier) {
      throw new Error('source model is not loaded');
    }
    const tfliteResults = this.tfliteNLClassifier.classify(text);
    if (!tfliteResults) {
      return {classes: []};
    }
    const classes: Class[] = tfliteResults.map(result => {
      return {
        className: result.className,
        score: result.probability,
      };
    });
    const finalResult: NLClassificationResult = {classes};
    return finalResult;
  }

  cleanUp() {
    if (!this.tfliteNLClassifier) {
      throw new Error('source model is not loaded');
    }
    this.tfliteNLClassifier.cleanUp();
  }
}

/** Merges the given options with the default NLClassifier options. */
export function getNLClassifierOptions(options?: tflite.NLClassifierOptions):
    tflite.NLClassifierOptions {
  const nlclassifierOptions: tflite.NLClassifierOptions = {
    inputTensorIndex: 0,
    outputScoreTensorIndex: 0,
    outputLabelTensorIndex: -1,
    inputTensorName: 'INPUT',
    outputScoreTensorName: 'OUTPUT_SCORE',
    outputLabelTensorName: 'OUTPUT_LABEL',
  };
  if (!options) {
    return nlclassifierOptions;
  }
  if (options.inputTensorIndex != null) {
    nlclassifierOptions.inputTensorIndex = options.inputTensorIndex;
  }
  if (options.outputScoreTensorIndex != null) {
    nlclassifierOptions.outputScoreTensorIndex = options.outputScoreTensorIndex;
  }
  if (options.outputLabelTensorIndex != null) {
    nlclassifierOptions.outputLabelTensorIndex = options.outputLabelTensorIndex;
  }
  if (options.inputTensorName != null) {
    nlclassifierOptions.inputTensorName = options.inputTensorName;
  }
  if (options.outputScoreTensorName != null) {
    nlclassifierOptions.outputScoreTensorName = options.outputScoreTensorName;
  }
  if (options.outputLabelTensorName != null) {
    nlclassifierOptions.outputLabelTensorName = options.outputLabelTensorName;
  }
  return nlclassifierOptions;
}
