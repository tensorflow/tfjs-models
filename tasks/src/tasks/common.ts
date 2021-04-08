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

/**
 * A transformer that loads an existing TFJS or TFLite task library model and
 * transforms it to a task model.
 *
 * <M>: The type of the transformed task model. Models with the same task type
 * will have the same task model. For example, TFLite image classifier model and
 * TFJS mobilenet model will be transformed to the same ImageClassifier task
 * model.
 *
 * <O>: The options to configure model loading and inference.
 */
export interface TaskModelTransformer<M, O> {
  loadAndTransform(options?: O): Promise<M>;
}
