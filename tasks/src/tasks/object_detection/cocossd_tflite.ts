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
import {ObjectDetectorTFLite} from './tflite_common';

// The global namespace type.
type TFLiteNS = typeof tflite;

/** Loading options. */
export interface CocoSsdTFLiteLoadingOptions extends
    tflite.ObjectDetectorOptions {}

/**
 * Inference options.
 *
 * TODO: placeholder for now.
 */
export interface CocoSsdTFLiteInferenceOptions {}

/** Loader for cocossd TFLite model. */
export class CocoSsdTFLiteLoader extends
    TaskModelLoader<TFLiteNS, CocoSsdTFLiteLoadingOptions, CocoSsdTFLite> {
  readonly metadata = {
    name: 'TFLite COCO-SSD',
    description: 'Run COCO-SSD object detection model with TFLite',
    runtime: Runtime.TFLITE,
    version: '0.0.1-alpha.3',
    supportedTasks: [Task.OBJECT_DETECTION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@${
          this.metadata.version}/dist/tf-tflite.min.js`]];
  readonly sourceModelGlobalNamespace = 'tflite';

  protected async transformSourceModel(
      sourceModelGlobal: TFLiteNS,
      loadingOptions?: CocoSsdTFLiteLoadingOptions): Promise<CocoSsdTFLite> {
    const url = 'https://tfhub.dev/tensorflow/lite-model/' +
        'ssd_mobilenet_v1/1/metadata/2?lite-format=tflite';
    const tfliteObjectDetector =
        await sourceModelGlobal.ObjectDetector.create(url, loadingOptions);
    return new CocoSsdTFLite(tfliteObjectDetector);
  }
}

/**
 * Pre-trained TFLite coco-ssd object detection model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * const model = await tfTask.ObjectDetection.CocoSsd.TFLite.load();
 *
 * // Run inference on an image.
 * const img = document.querySelector('img');
 * const result = await model.predict(img);
 * console.log(result.objects);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.ObjectDetector` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol:
 * 'CocoSsdTFLiteLoadingOptions'},
 *   {description: 'Options for `predict`',
 * symbol: 'CocoSsdTFLiteInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Object Detection', subheading: 'Models'}
 */
export class CocoSsdTFLite extends
    ObjectDetectorTFLite<CocoSsdTFLiteInferenceOptions> {}

export const cocoSsdTfliteLoader = new CocoSsdTFLiteLoader();
