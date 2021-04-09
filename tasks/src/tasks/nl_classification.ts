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

import {NLClassifierOptions} from '@tensorflow/tfjs-tflite';
import * as tfwebClient from '../tfweb_client';

import {BaseTaskModel, TaskModelTransformer} from './common';

/**
 * The canonical NLClassifier task model for all NL classification realted
 * TFJS/TFLite models
 */
export interface NLClassifier extends BaseTaskModel {
  /** Performs classification on the given text, and returns result. */
  classify(text: string): Promise<NLClassifierResult>;
}

/** Classification result. */
export interface NLClassifierResult {
  categories: Category[];
}

/** A single classification category. */
export interface Category {
  className: string;
  score: number;
}

/** Base options for all TFLite models. */
export interface TFLiteBaseOptions {
  // Options for tfweb.
  tfwebNLClassifierOptions?: NLClassifierOptions;
}

/** Options for TFLite NL classification custom models. */
export interface TFLiteNLClassificationCustomModelOptions extends
    TFLiteBaseOptions {
  modelUrl: string;
}

/** Default tfweb NLClassifier options. */
const DEFAULT_NLCLASSIFIER_OPTIONS: NLClassifierOptions = {
  inputTensorIndex: 0,
  outputScoreTensorIndex: 0,
  outputLabelTensorIndex: -1,
  inputTensorName: 'INPUT',
  outputScoreTensorName: 'OUTPUT_SCORE',
  outputLabelTensorName: 'OUTPUT_LABEL',
};

/** The transformer for the TFLite NLClassifier custom model url. */
export const TFLiteNLClassifierTransformer: TaskModelTransformer<
    NLClassifier, TFLiteNLClassificationCustomModelOptions> = {
  async loadAndTransform(options?: TFLiteNLClassificationCustomModelOptions):
      Promise<NLClassifier> {
        if (!options) {
          throw new Error('Must provide options with modelUrl specified');
        }
        return transformTFLiteNLClassifierModel(options.modelUrl, options);
      }
}

async function transformTFLiteNLClassifierModel(
    modelUrl: string, options?: TFLiteBaseOptions): Promise<NLClassifier> {
  // Load the model.
  const tfwebOptions = (options && options.tfwebNLClassifierOptions) ||
      DEFAULT_NLCLASSIFIER_OPTIONS;
  const tfwebNLClassifier =
      await tfwebClient.tfweb.NLClassifier.create(modelUrl, tfwebOptions);

  // Transform to NLClassifier task model.
  const nlClassifier: NLClassifier = {
    classify: async (text) => {
      const tfwebResults = tfwebNLClassifier.classify(text);
      const categories: Category[] = [];
      if (tfwebResults) {
        categories.push(...tfwebResults.map(tfwebCategory => {
          return {
            className: tfwebCategory.className,
            score: tfwebCategory.score,
          };
        }));
      }
      const finalResult: NLClassifierResult = {
        categories,
      };
      return finalResult;
    },

    cleanUp: () => {
      tfwebNLClassifier.cleanUp();
    },
  };
  return nlClassifier;
}
