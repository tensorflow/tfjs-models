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
import {TaskModelLoader} from '../../task_model';
import {ensureTFJSBackend, Runtime, Task, TFJSModelCommonLoadingOption} from '../common';
import {Class, ImageClassifier, ImageClassifierResult} from './common';

// The global namespace type.
type MobilenetNS = typeof mobilenet;

/** Loading options. */
export type MobilenetTFJSLoadingOptions =
    TFJSModelCommonLoadingOption&mobilenet.ModelConfig;

/** Inference options. */
export interface MobilenetTFJSInferenceOptions {
  /** Number of top classes to return. */
  topK?: number;
}

/** Loader for mobilenet TFJS model. */
export class MobilenetTFJSLoader extends TaskModelLoader<
    MobilenetNS, MobilenetTFJSLoadingOptions,
    ImageClassifier<MobilenetTFJSInferenceOptions>> {
  readonly metadata = {
    name: 'TFJS Mobilenet',
    description: 'Run mobilenet with TFJS models',
    resourceUrls: {
      'github':
          'https://github.com/tensorflow/tfjs-models/tree/master/mobilenet',
    },
    runtime: Runtime.TFJS,
    version: '2.1.0',
    supportedTasks: [Task.IMAGE_CLASSIFICATION],
  };
  readonly packageUrls =
      [[`https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@${
          this.metadata.version}`]];
  readonly sourceModelGlobalNamespace = 'mobilenet';

  protected async transformSourceModel(
      sourceModelGlobal: MobilenetNS,
      loadingOptions?: MobilenetTFJSLoadingOptions):
      Promise<ImageClassifier<MobilenetTFJSInferenceOptions>> {
    const mobilenetModel = await sourceModelGlobal.load(loadingOptions);

    return {
      predict: async (img, infereceOptions) => {
        if (!mobilenetModel) {
          throw new Error('source model is not loaded');
        }
        await ensureTFJSBackend(loadingOptions);
        const mobilenetResults = await mobilenetModel.classify(
            img, infereceOptions ? infereceOptions.topK : undefined);
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
      },

      cleanUp: () => {},
    };
  }
}

export const mobilenetTfjsLoader = new MobilenetTFJSLoader();
