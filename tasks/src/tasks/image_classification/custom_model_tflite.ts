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
import {Runtime, Task, TFLiteCustomModelCommonLoadingOption} from '../common';
import {ImageClassifier} from './common';
import {transformImageClassifier} from './tflite_utils';

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

/** Loader for custom image classification TFLite model. */
export class ImageClassificationCustomModelTFLiteLoader extends TaskModelLoader<
    TFLiteNS, ImageClassificationCustomModelTFLiteLoadingOptions,
    ImageClassifier<ImageClassificationCustomModelTFLiteInferanceOptions>> {
  readonly metadata = {
    name: 'Image classification with TFLite models',
    description: 'An image classfier backed by the TFLite Task Library. ' +
        'It can work with any models that meet the ' +
        '<a href:"https://www.tensorflow.org/lite/inference_with_metadata/' +
        'task_library/image_classifier#model_compatibility_requirements" ' +
        'target:"_blank">model requirements</a>.',
    resourceUrls: {
      'TFLite task library': 'https://www.tensorflow.org/lite/' +
          'inference_with_metadata/task_library/overview',
    },
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.IMAGE_CLASSIFICATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNs = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: ImageClassificationCustomModelTFLiteLoadingOptions):
      Promise<ImageClassifier<
          ImageClassificationCustomModelTFLiteInferanceOptions>> {
    return transformImageClassifier(sourceModelGlobal, loadingOptions.model);
  }
}

export const imageClassificationCustomModelTfliteLoader =
    new ImageClassificationCustomModelTFLiteLoader();
