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

import {TaskModel} from '../../task_model';
import {Task} from '../common';

/**
 * The common base class for all ImageClassification task models.
 *
 * @template N The global namespace provided by the source model.
 * @template LO The type of options used during the model loading process.
 * @template IO The type of options used during the inference process.
 */
export abstract class ImageClassifier<N, LO, IO> extends TaskModel<N, LO> {
  readonly supportedTasks = [Task.IMAGE_CLASSIFICATION];

  /**
   * Performs classification on the given image-like input, and returns
   * result.
   */
  abstract classify(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      options?: IO): Promise<ImageClassifierResult>;
}

/** Image classification result. */
export interface ImageClassifierResult {
  classes: Class[];
}

/** A single class in the classification result. */
export interface Class {
  className: string;
  probability: number;
}
