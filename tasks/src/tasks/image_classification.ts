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
import * as tfwebClient from '../tfweb_client';

import {BaseTaskModel, TaskModelTransformer} from './common';

/**
 * The canonical ImageClassifier task model for all image classification related
 * TFJS/TFLite models.
 */
export interface ImageClassifier extends BaseTaskModel {
  /**
   * Performs classification on the given image-like resource, and returns
   * result.
   */
  classify(img: ImageData|HTMLImageElement|HTMLCanvasElement|
           HTMLVideoElement): Promise<ImageClassifierResult>;
}

/** Classification result. */
export interface ImageClassifierResult {
  classes: Class[];
}

/** A single class in the classification result. */
export interface Class {
  className: string;
  probability: number;
}

/** Options for TFJS mobilenet model. */
export interface TFJSMobileNetOptoins {
  maxResult?: number;
  modelConfig?: mobilenet.ModelConfig;
}

/** Base options for all TFLite models. */
export interface TFLiteBaseOptions {
  maxResults?: number;
  scoreThreshold?: number;
  numThreads?: number;
}

/** Options for TFLiteMobileNet models. */
export interface TFLiteMobileNetOptions extends TFLiteBaseOptions {
  version?: 1|2;
  alpha?: 0.25|0.50|0.75|1.0;
}

/** Options for TFLite image classification custom models. */
export interface TFLiteImageClassificationCustomModelOptions extends
    TFLiteBaseOptions {
  modelUrl: string;
}

/** The transformer for the TFJS MobileNet model. */
export const TFJSMobileNetTransformer:
    TaskModelTransformer<ImageClassifier, TFJSMobileNetOptoins> = {
      async loadAndTransform(options?: TFJSMobileNetOptoins):
          Promise<ImageClassifier> {
            // Load the model.
            if (!mobilenet) {
              throw new Error('@tensorflow-models/mobilenet package not found');
            }
            let config: mobilenet.ModelConfig =
                (options && options.modelConfig) || {version: 1, alpha: 1.0};
            const mobilenetModel = await mobilenet.load(config);

            // Transform to ImageClassifier task model.
            const imageClassifier: ImageClassifier = {
              classify: async (img) => {
                const mobilenetResults = await mobilenetModel.classify(
                    img, options ? options.maxResult : undefined);
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

            return imageClassifier;
          }

    };

/** The transformer for the TFLite MobileNet model. */
export const TFLiteMobileNetTransformer:
    TaskModelTransformer<ImageClassifier, TFLiteMobileNetOptions> = {
      async loadAndTransform(options?: TFLiteMobileNetOptions):
          Promise<ImageClassifier> {
            let modelName = '';
            const version = `${((options && options.version) || 1).toFixed(2)}`;
            const alpha = `${((options && options.alpha) || 1.0).toFixed(2)}`;
            if (version === '1.00') {
              if (alpha === '0.25') {
                modelName = 'mobilenet_v1_0.25_224_1_metadata_1.tflite';
              } else if (alpha === '0.50') {
                modelName = 'mobilenet_v1_0.50_224_1_metadata_1.tflite';
              } else if (alpha === '0.75') {
                modelName = 'mobilenet_v1_0.75_224_1_metadata_1.tflite';
              } else if (alpha === '1.00') {
                modelName = 'mobilenet_v1_1.0_224_1_metadata_1.tflite';
              }
            } else if (version === '2.00') {
              if (alpha === '1.00') {
                modelName = 'mobilenet_v2_1.0_224_1_metadata_1.tflite';
              } else {
                modelName = 'mobilenet_v1_1.0_224_1_metadata_1.tflite';
                console.warn(`WARNING: MobileNet v${version}_${
                    alpha} not supported. Use the default one instead: ${
                    modelName}`);
              }
            }
            // TODO: use tfhub url instead when it is ready.
            const modelUrl =
                `https://storage.googleapis.com/tfweb/models/${modelName}`;
            return transformTFLiteImageClassifierModel(modelUrl, options);
          }
    };

/** The transformer for the TFLite image classifier with custom model url. */
export const TFLiteImageClassifierTransformer: TaskModelTransformer<
    ImageClassifier, TFLiteImageClassificationCustomModelOptions> = {
  async loadAndTransform(options?: TFLiteImageClassificationCustomModelOptions):
      Promise<ImageClassifier> {
        if (!options) {
          throw new Error('Must provide options with modelUrl specified');
        }
        return transformTFLiteImageClassifierModel(options.modelUrl, options);
      }
};

async function transformTFLiteImageClassifierModel(
    modelUrl: string, options?: TFLiteBaseOptions): Promise<ImageClassifier> {
  // Load the model.
  const tfwebOptions = new tfwebClient.tfweb.ImageClassifierOptions();
  if (options) {
    if (options.maxResults !== undefined) {
      tfwebOptions.setMaxResults(options.maxResults);
    }
    if (options.numThreads !== undefined) {
      tfwebOptions.setNumThreads(options.numThreads);
    }
    if (options.scoreThreshold !== undefined) {
      tfwebOptions.setScoreThreshold(options.scoreThreshold);
    }
  }
  const tfwebImageClassifier =
      await tfwebClient.tfweb.ImageClassifier.create(modelUrl, tfwebOptions);

  // Transform to ImageClassifier task model.
  const imageClassifier: ImageClassifier = {
    classify: async (img) => {
      const tfwebResults = tfwebImageClassifier.classify(img);
      const classes: Class[] = [];
      const tfwebClassificationsList = tfwebResults.getClassificationsList();
      if (tfwebClassificationsList.length > 0) {
        tfwebClassificationsList[0].getClassesList().forEach(cls => {
          classes.push({
            className: cls.getDisplayName() || cls.getClassName(),
            probability: cls.getScore(),
          });
        });
      }
      const finalResult: ImageClassifierResult = {
        classes,
      };
      return finalResult;
    },

    cleanUp: () => {
      tfwebImageClassifier.cleanUp();
    },
  };

  return imageClassifier;
}
