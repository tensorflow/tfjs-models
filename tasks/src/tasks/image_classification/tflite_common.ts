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

import {ImageClassificationResult, ImageClassifier} from './common';

/**
 * The base class for all image classification TFLite models.
 *
 * @template T The type of inference options.
 */
export class ImageClassifierTFLite<T> extends ImageClassifier<T> {
  constructor(private tfliteImageClassifier: tflite.ImageClassifier) {
    super();
  }

  async predict(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      infereceOptions?: T): Promise<ImageClassificationResult> {
    if (!this.tfliteImageClassifier) {
      throw new Error('source model is not loaded');
    }
    const tfliteResults = this.tfliteImageClassifier.classify(img);
    if (!tfliteResults) {
      return {classes: []};
    }
    const classes: Class[] = tfliteResults.map(result => {
      return {
        className: result.className,
        score: result.probability,
      };
    });
    const finalResult: ImageClassificationResult = {classes};
    return finalResult;
  }

  cleanUp() {
    if (!this.tfliteImageClassifier) {
      throw new Error('source model is not loaded');
    }
    this.tfliteImageClassifier.cleanUp();
  }
}
