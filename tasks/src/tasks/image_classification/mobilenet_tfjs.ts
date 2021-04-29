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

import * as mobilenet from '@tensorflow-models/mobilenet';
import {Runtime, TFJSModelCommonLoadingOption} from '../common';
import {Class, ImageClassifier, ImageClassifierResult} from './common';

// The global namespace type.
type MobilenetNS = typeof mobilenet;

/** Loading options. */
export type MobilenetTFJSLoadOptions =
    TFJSModelCommonLoadingOption&mobilenet.ModelConfig;

/** Inference options. */
export interface MobilenetTFJSInferenceOptions {
  /** Number of top classes to return. */
  topK?: number;
}

/** Mobilenet model from TFJS. */
export class MobilenetTFJS extends ImageClassifier<
    MobilenetNS, MobilenetTFJSLoadOptions, MobilenetTFJSInferenceOptions> {
  private mobilenetModel: mobilenet.MobileNet;

  readonly name = 'Mobilenet';
  readonly runtime = Runtime.TFJS;
  readonly version = '2.1.0';
  readonly packageUrls =
      [['https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0']];
  readonly sourceModelGlobalNs = 'mobilenet';

  protected async loadSourceModel(
      sourceModelGlobal: MobilenetNS, options?: MobilenetTFJSLoadOptions) {
    this.mobilenetModel = await sourceModelGlobal.load(options);
  }

  async classify(
      img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      options?: MobilenetTFJSInferenceOptions): Promise<ImageClassifierResult> {
    if (!this.mobilenetModel) {
      throw new Error('source model is not loaded');
    }
    const mobilenetResults = await this.mobilenetModel.classify(
        img, options ? options.topK : undefined);
    const classes: Class[] = mobilenetResults.map(result => {
      return {
        className: result.className,
        probability: result.probability,
      };
    });
    const finalResult: ImageClassifierResult = {
      classes,
    };
    return finalResult;
  }
}

export const mobilenetTfjs = new MobilenetTFJS();
