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
import {TaskModelLoader} from '../../task_model';
import {Runtime, Task} from '../common';
import {ImageClassifier} from './common';
import {transformImageClassifier} from './tflite_utils';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export type MobilenetTFLiteLoadingOptions = tflite.ImageClassifierOptions&{
  version?: 1|2;
  alpha?: 0.25|0.50|0.75|1.0;
};

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export type MobilenetTFLiteInferanceOptions = {}

/** Loader for custom image classification TFLite model. */
export class MobilenetTFLiteLoader extends TaskModelLoader<
    TFLiteNS, MobilenetTFLiteLoadingOptions,
    ImageClassifier<MobilenetTFLiteInferanceOptions>> {
  readonly name = 'TFLite Mobilenet';
  readonly description = 'Run mobilenet with TFLite models';
  readonly runtime = Runtime.TFLITE;
  readonly version = '0.0.1-alpha.3';
  readonly supportedTasks = [Task.IMAGE_CLASSIFICATION];
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNs = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: MobilenetTFLiteLoadingOptions):
      Promise<ImageClassifier<MobilenetTFLiteInferanceOptions>> {
    // Construct the mobilenet model url from version and alpha.
    let mobilenetVersion = '2';
    let mobilenetAlpha = '1.0';
    if (loadingOptions) {
      if (loadingOptions.version !== undefined) {
        mobilenetVersion = String(loadingOptions.version);
      }
      if (loadingOptions.alpha !== undefined) {
        mobilenetAlpha = String(loadingOptions.alpha);
      }
    }
    // There is no TFLite mobilenet v2 model available other than 2_1.0.
    if (mobilenetVersion === '2' && mobilenetAlpha !== '1.0') {
      mobilenetAlpha = '1.0';
      console.warn(`No mobilenet TFLite model available for ${
          mobilenetVersion}_${mobilenetAlpha}. Using 2_1.0 instead.`);
    }
    // TODO: use TFHub url when CORS is correctly set.
    const url = `https://storage.googleapis.com/tfweb/models/mobilenet_v${
        mobilenetVersion}_${mobilenetAlpha}_224_1_metadata_1.tflite`;

    return transformImageClassifier(sourceModelGlobal, url);
  }
}

export const mobilenetTfliteLoader = new MobilenetTFLiteLoader();
