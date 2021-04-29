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
import {Runtime, TFLiteCustomModelCommonLoadingOption} from '../common';
import {Class, ImageClassifier, ImageClassifierResult} from './common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export type ImageClassificationCustomModelTFLiteLoadingOptions =
    TFLiteCustomModelCommonLoadingOption&tflite.ImageClassifierOptions;

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export type ImageClassificationCustomModelTFLiteInferanceOptions = {}

/** Custom TFLite model. */
export class ImageClassificationCustomModelTFLite extends ImageClassifier<
    TFLiteNS, ImageClassificationCustomModelTFLiteLoadingOptions,
    ImageClassificationCustomModelTFLiteInferanceOptions> {
  private tfliteImageClassifier: tflite.ImageClassifier;

  readonly name = 'Custom image classification TFLite model';
  readonly runtime = Runtime.TFLITE;
  readonly version = '0.0.1-alpha.3';
  readonly packageUrls =
      [['https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.3' +
        '/dist/tf-tflite.min.js']];
  readonly sourceModelGlobalNs = 'tflite';

  protected async loadSourceModel(
      sourceModelGlobal: TFLiteNS,
      options?: ImageClassificationCustomModelTFLiteLoadingOptions) {
    this.tfliteImageClassifier =
        await sourceModelGlobal.ImageClassifier.create(options.model);
  }

  async classify(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      options?: ImageClassificationCustomModelTFLiteInferanceOptions):
      Promise<ImageClassifierResult> {
    if (!this.tfliteImageClassifier) {
      throw new Error('source model is not loaded');
    }
    const tfliteResults = this.tfliteImageClassifier.classify(img);
    const classes: Class[] = tfliteResults.map(result => {
      return {
        className: result.className,
        probability: result.probability,
      };
    });
    const finalResult: ImageClassifierResult = {
      classes,
    };
    return finalResult;
  }

  cleanUp() {
    if (!this.tfliteImageClassifier) {
      throw new Error('source model is not loaded');
    }
    this.tfliteImageClassifier.cleanUp();
  }
}

export const imageClassificationCustomModelTflite =
    new ImageClassificationCustomModelTFLite();
