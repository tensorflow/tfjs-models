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

import {TaskModelTransformer} from './tasks/common';

/**
 * The main function to load a task model.
 *
 * @param taskModelTransforer The transformer for the task model. Typically this
 *     is specified by using `ML_TASK` in tasks/all_tasks.ts (e.g.
 *     ML_TASK.ImageClassifier_TFLite).
 * @param options The options to configure model loading and inference.
 */
export async function loadTaskModel<M, O>(
    taskModelTransformer: TaskModelTransformer<M, O>, options?: O): Promise<M> {
  return taskModelTransformer.loadAndTransform(options);
}
