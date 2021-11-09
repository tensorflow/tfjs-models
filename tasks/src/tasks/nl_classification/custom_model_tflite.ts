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
import {getNLClassifierOptions, NLClassifierTFLite} from './tflite_common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export interface NCCustomModelTFLiteLoadingOptions extends
    TFLiteCustomModelCommonLoadingOption, tflite.NLClassifierOptions {}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface NCCustomModelTFLiteInferenceOptions {}

/** Loader for custom nl classification TFLite model. */
export class NLClassificationCustomModelTFLiteLoader extends TaskModelLoader<
    TFLiteNS, NCCustomModelTFLiteLoadingOptions, NCCustomModelTFLite> {
  readonly metadata = {
    name: 'Natural language classification with TFLite models',
    description: 'A natural language detector backed by the NLClassifier in ' +
        'TFLite Task Library. ' +
        'It can work with any models that meet the ' +
        '<a href="https://www.tensorflow.org/lite/inference_with_metadata/' +
        'task_library/nl_classifier#model_compatibility_requirements" ' +
        'target="_blank">model requirements</a>.',
    resourceUrls: {
      'TFLite task library': 'https://www.tensorflow.org/lite/' +
          'inference_with_metadata/task_library/overview',
    },
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.NL_CLASSIFICATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: NCCustomModelTFLiteLoadingOptions):
      Promise<NCCustomModelTFLite> {
    const tfliteNLClassifier = await sourceModelGlobal.NLClassifier.create(
        loadingOptions.model, getNLClassifierOptions(loadingOptions));
    return new NCCustomModelTFLite(tfliteNLClassifier);
  }
}

/**
 * A custom TFLite natural language classification model loaded from a model url
 * or an `ArrayBuffer` in memory.
 *
 * The underlying NL classifier is built on top of the NLClassifier in
 * [TFLite Task
 * Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview).
 * As a result, the custom model needs to meet the [metadata
 * requirements](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier#model_compatibility_requirements).
 *
 * Usage:
 *
 * ```js
 * // Load the model from a custom url with other options (optional).
 * const model = await tfTask.NLClassification.CustomModel.TFLite.load({
 *   model:
 * 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite',
 * });
 *
 * // Run inference on text.
 * const result = await model.predict('This is a great movie!');
 * console.log(result.classes);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.NLClassifier` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol:
 * 'NCCustomModelTFLiteLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'NCCustomModelTFLiteInferenceOptions'}
 * ]
 *
 *
 * @doc {heading: 'NL Classification', subheading: 'Models'}
 */
export class NCCustomModelTFLite extends
    NLClassifierTFLite<NCCustomModelTFLiteInferenceOptions> {}

export const nlClassifierCustomModelTfliteLoader =
    new NLClassificationCustomModelTFLiteLoader();
