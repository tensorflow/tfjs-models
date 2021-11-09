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

import * as cocoSsd from '@tensorflow-models/coco-ssd';

import {TaskModelLoader} from '../../task_model';
import {ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';

import {DetectedObject, ObjectDetectionResult, ObjectDetector} from './common';

// The global namespace type.
type CocoSsdNS = typeof cocoSsd;

/** Loading options. */
export interface CocoSsdTFJSLoadingOptions extends TFJSModelCommonLoadingOption,
                                                   cocoSsd.ModelConfig {}

/** Inference options. */
export interface CocoSsdTFJSInferenceOptions {
  /**
   * The maximum number of bounding boxes of detected objects. There can be
   * multiple objects of the same class, but at different locations. Defaults
   * to 20.
   */
  maxNumBoxes?: number;
  /**
   * The minimum score of the returned bounding boxes of detected objects. Value
   * between 0 and 1. Defaults to 0.5.
   */
  minScore?: number;
}

/** Loader for cocossd TFJS model. */
export class CocoSsdTFJSLoader extends
    TaskModelLoader<CocoSsdNS, CocoSsdTFJSLoadingOptions, CocoSsdTFJS> {
  readonly metadata = {
    name: 'TFJS COCO-SSD',
    description: 'Run COCO-SSD object detection model with TFJS',
    resourceUrls: {
      'github':
          'https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd',
    },
    runtime: Runtime.TFJS,
    version: '2.2.2',
    supportedTasks: [Task.OBJECT_DETECTION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'cocoSsd';

  protected async transformSourceModel(
      sourceModelGlobal: CocoSsdNS,
      loadingOptions?: CocoSsdTFJSLoadingOptions): Promise<CocoSsdTFJS> {
    const cocoSsdModel = await sourceModelGlobal.load(loadingOptions);
    return new CocoSsdTFJS(cocoSsdModel, loadingOptions);
  }
}

/**
 * Pre-trained TFJS coco-ssd model.
 *
 * Usage:
 *
 * ```js
 * // Load the model with options (optional).
 * //
 * // By default, it uses lite_mobilenet_v2 as the base model with webgl
 * // backend. You can change them in the `options` parameter of the `load`
 * // function (see below for docs).
 * const model = await tfTask.ObjectDetection.CocoSsd.TFJS.load();
 *
 * // Run detection on an image with options (optional).
 * const img = document.querySelector('img');
 * const result = await model.predict(img, {numMaxBoxes: 5});
 * console.log(result.objects);
 *
 * // Clean up.
 * model.cleanUp();
 * ```
 *
 * Refer to `tfTask.ObjectDetector` for the `predict` and `cleanUp` method.
 *
 * @docextratypes [
 *   {description: 'Options for `load`', symbol: 'CocoSsdTFJSLoadingOptions'},
 *   {description: 'Options for `predict`', symbol:
 * 'CocoSsdTFJSInferenceOptions'}
 * ]
 *
 * @doc {heading: 'Object Detection', subheading: 'Models'}
 */
export class CocoSsdTFJS extends ObjectDetector<CocoSsdTFJSInferenceOptions> {
  constructor(
      private cocoSsdModel?: cocoSsd.ObjectDetection,
      private loadingOptions?: CocoSsdTFJSLoadingOptions) {
    super();
  }

  async predict(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      infereceOptions?: CocoSsdTFJSInferenceOptions):
      Promise<ObjectDetectionResult> {
    if (!this.cocoSsdModel) {
      throw new Error('source model is not loaded');
    }
    await ensureTFJSBackend(this.loadingOptions);
    const cocoSsdResults = await this.cocoSsdModel.detect(
        img, infereceOptions ? infereceOptions.maxNumBoxes : undefined,
        infereceOptions ? infereceOptions.minScore : undefined);
    const objects: DetectedObject[] = cocoSsdResults.map(result => {
      return {
        boundingBox: {
          originX: result.bbox[0],
          originY: result.bbox[1],
          width: result.bbox[2],
          height: result.bbox[3],
        },
        className: result.class,
        score: result.score,
      };
    });
    const finalResult: ObjectDetectionResult = {
      objects,
    };
    return finalResult;
  }

  cleanUp() {
    if (!this.cocoSsdModel) {
      throw new Error('source model is not loaded');
    }
    return this.cocoSsdModel.dispose();
  }
}

export const cocoSsdTfjsLoader = new CocoSsdTFJSLoader();
