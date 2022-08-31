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

import {DetectedObject, ObjectDetectionResult, ObjectDetector} from './common';

/**
 * The base class for all object detection TFLite models.
 *
 * @template T The type of inference options.
 */
export class ObjectDetectorTFLite<T> extends ObjectDetector<T> {
  constructor(private tfliteObjectDetector: tflite.ObjectDetector) {
    super();
  }

  async predict(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      infereceOptions?: T): Promise<ObjectDetectionResult> {
    if (!this.tfliteObjectDetector) {
      throw new Error('source model is not loaded');
    }
    const tfliteResults = this.tfliteObjectDetector.detect(img);
    if (!tfliteResults) {
      return {objects: []};
    }
    const objects: DetectedObject[] = tfliteResults.map(result => {
      return {
        boundingBox: result.boundingBox,
        className: result.classes[0].className,
        score: result.classes[0].probability,
      };
    });
    const finalResult: ObjectDetectionResult = {objects};
    return finalResult;
  }

  cleanUp() {
    if (!this.tfliteObjectDetector) {
      throw new Error('source model is not loaded');
    }
    this.tfliteObjectDetector.cleanUp();
  }
}
