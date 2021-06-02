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

import {TaskModelLoader} from '../task_model';
import {modelIndex} from './all_tasks';
import {Runtime, Task} from './common';

describe('model index', () => {
  it('should index models at the right places', () => {
    // Iterate through all model loders and check that their runtime and tasks
    // match the location.
    for (const task of Object.keys(modelIndex)) {
      // tslint:disable-next-line:no-any
      const models = (modelIndex as any)[task];
      for (const modelName of Object.keys(models)) {
        const runtimes = models[modelName];
        for (const runtime of Object.keys(runtimes)) {
          const modelLoader = runtimes[runtime] as TaskModelLoader<{}, {}, {}>;
          // Check runtime.
          expect(modelLoader.metadata.runtime).toEqual(runtime as Runtime);
          // Check task label.
          expect(modelLoader.metadata.supportedTasks.includes(task as Task))
              .toBe(true);
        }
      }
    }
  });
});
