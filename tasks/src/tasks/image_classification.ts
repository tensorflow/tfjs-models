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

import {TaskModelTransformer} from './common';

/**
 * The canonical ImageClassifier task model for all image classification related
 * TFJS/TFLite models.
 */
export interface ImageClassifier {
  /**
   * Performs classification on the given image-like resource, and returns
   * result.
   */
  classify(img: ImageData|HTMLImageElement|HTMLCanvasElement|
           HTMLVideoElement): Promise<ImageClassifierResult>;
}

/** Options to customize model loading and inference. */
export interface ImageClassifierOptions {
  // Common options.
  maxResult?: number;

  // Options for tfweb.
  modelPath?: string;
  scoreThreshold?: number;
  numThreads?: number;

  // Options for tfjs mobilenet model.
  mobilentConfig?: mobilenet.ModelConfig;
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

/** The transformer for the TFJS mobilenet model. */
export const ImageClassifierTFJS:
    TaskModelTransformer<ImageClassifier, ImageClassifierOptions> = {
      async loadAndTransform(options?: ImageClassifierOptions):
          Promise<ImageClassifier> {
            // Load the model.
            if (!mobilenet) {
              throw new Error('@tensorflow-models/mobilenet package not found');
            }
            const mobilenetModel = await mobilenet.load(
                options ? options.mobilentConfig : undefined);

            // Transform to ImageClassifier.
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
            };

            return imageClassifier;
          }
    };

/** The transformer for the tfweb image classifier. */
export const ImageClassifierTFLite:
    TaskModelTransformer<ImageClassifier, ImageClassifierOptions> = {
      async loadAndTransform(options?: ImageClassifierOptions):
          Promise<ImageClassifier> {
            if (!options || !options.modelPath) {
              throw new Error('Must provide modelPath');
            }
            // Load the model.
            const tfwebOptions = new tfwebClient.tfweb.ImageClassifierOptions();
            if (options.maxResult) {
              tfwebOptions.setMaxResults(options.maxResult);
            }
            if (options.scoreThreshold) {
              tfwebOptions.setScoreThreshold(options.scoreThreshold);
            }
            if (options.numThreads) {
              tfwebOptions.setNumThreads(options.numThreads);
            }
            const tfwebImageClassifier =
                await tfwebClient.tfweb.ImageClassifier.create(
                    options.modelPath, tfwebOptions);

            // Transform to ImageClassifier.
            const imageClassifier: ImageClassifier = {
              classify: async (img) => {
                const tfwebResults = tfwebImageClassifier.classify(img);
                const classes: Class[] = [];
                const tfwebClassificationsList =
                    tfwebResults.getClassificationsList();
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
            };

            return imageClassifier;
          }
    };
