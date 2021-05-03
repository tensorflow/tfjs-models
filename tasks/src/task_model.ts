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

import {BaseTaskModel, TaskModelTransformer} from './tasks/common';

/**
 * The main function to load a task model.
 *
 * <M>: The type of the transformed task model. Models with the same task type
 *     will have the same task model. For example, TFLite image classifier model
 *     and TFJS mobilenet model will be transformed to the same ImageClassifier
 *     task model.
 *
 * <O>: The type of options to configure model loading and inference. Different
 *     models could be associated with different option types.
 *
 * Sample usage:
 *
 * // Use a pre-trained TFLite mobilenet model.
 * const model = await loadTaskModel(
 *     MLTask.ImageClassification.TFLiteMobileNet);
 *
 * // Use a custom TFLite image classification model.
 * const model = await loadTaskModel(
 *     MLTask.ImageClassification.TFLiteCustomModel,
 *     {modelUrl: 'url/to/your/model.tflite'});
 *
 * // Use a pre-trained TFJS mobilenet model.
 * const model = await loadTaskModel(
 *     MLTask.ImageClassification.TFJSMobileNet,
 *     {version: 2, alpha: 1.0});
 *
 * // Task models for the same task type (MLTask.ImageClassification in this
 * // case) have the same model type (ImageClassifier in this case) that provide
 * // methods best and most intuitive for the task.
 * const result = model.classify(img)
 * console.log(result.classes);
 *
 * @param taskModelTransformer The transformer for the task model. Typically
 *     this is specified by using `MLTask` in tasks/all_tasks.ts (e.g.
 *     MLTask.ImageClassification.TFJSMobileNet).
 * @param options The options to configure model loading and inference.
 *
 * @returns The transformed task model.
 */
export async function loadTaskModel<M extends BaseTaskModel, O>(
    taskModelTransformer: TaskModelTransformer<M, O>, options?: O): Promise<M> {
  return taskModelTransformer.loadAndTransform(options);
}
