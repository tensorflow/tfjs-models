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

import {Runtime, Task} from './common';
import {imageClassificationCustomModelTfliteLoader} from './image_classification/custom_model_tflite';
import {mobilenetTfjsLoader} from './image_classification/mobilenet_tfjs';
import {mobilenetTfliteLoader} from './image_classification/mobilenet_tflite';
import {imageSegmenterCustomModelTfliteLoader} from './image_segmentation/custom_model_tflite';
import {deeplabTfjsLoader} from './image_segmentation/deeplab_tfjs';
import {deeplabTfliteLoader} from './image_segmentation/deeplab_tflite';
import {nlClassifierCustomModelTfliteLoader} from './nl_classification/custom_model_tflite';
import {cocoSsdTfjsLoader} from './object_detection/cocossd_tfjs';
import {cocoSsdTfliteLoader} from './object_detection/cocossd_tflite';
import {objectDetectorCustomModelTfliteLoader} from './object_detection/custom_model_tflite';
import {bertQaTfjsLoader} from './question_and_answer/bert_qa_tfjs';
import {bertQaTfliteLoader} from './question_and_answer/bert_qa_tflite';
import {qaCustomModelTfliteLoader} from './question_and_answer/custom_model_tflite';
import {movieReviewTfliteLoader} from './sentiment_detection/movie_review_tflite';
import {toxicityTfjsLoader} from './sentiment_detection/toxicity_tfjs';

/**
 * The main model index.
 *
 * The index structure is: {task_name}.{model_name}.{runtime}
 *
 * Note that it is possible to programmatically generate the index from a list
 * of loaders, but it would mean that we need to define a generic type for
 * each level of the index structure (e.g. {[taskName: string]: TaskModels}).
 * This will not work well for the auto-complete system in IDEs because
 * typescript doesn't know the candidates to show from the generic types.
 *
 * For this reason, we chose to create the index manually like below. In this
 * way, typescript will infer the types based on the content statically, and
 * correctly show the candidates when doing auto-complete.
 *
 * TODO: add test to make sure loaders are located in the correct index entries.
 */
export const modelIndex = {
  [Task.IMAGE_CLASSIFICATION]: {
    MobileNet: {
      [Runtime.TFJS]: mobilenetTfjsLoader,
      [Runtime.TFLITE]: mobilenetTfliteLoader,
    },
    CustomModel: {
      [Runtime.TFLITE]: imageClassificationCustomModelTfliteLoader,
    },
  },
  [Task.OBJECT_DETECTION]: {
    CocoSsd: {
      [Runtime.TFJS]: cocoSsdTfjsLoader,
      [Runtime.TFLITE]: cocoSsdTfliteLoader,
    },
    CustomModel: {
      [Runtime.TFLITE]: objectDetectorCustomModelTfliteLoader,
    },
  },
  [Task.IMAGE_SEGMENTATION]: {
    Deeplab: {
      [Runtime.TFJS]: deeplabTfjsLoader,
      [Runtime.TFLITE]: deeplabTfliteLoader,
    },
    CustomModel: {
      [Runtime.TFLITE]: imageSegmenterCustomModelTfliteLoader,
    },
  },
  [Task.SENTIMENT_DETECTION]: {
    MovieReview: {
      [Runtime.TFLITE]: movieReviewTfliteLoader,
    },
    Toxicity: {
      [Runtime.TFJS]: toxicityTfjsLoader,
    }
  },
  [Task.NL_CLASSIFICATION]: {
    CustomModel: {
      [Runtime.TFLITE]: nlClassifierCustomModelTfliteLoader,
    },
  },
  [Task.QUESTION_AND_ANSWER]: {
    BertQA: {
      [Runtime.TFJS]: bertQaTfjsLoader,
      [Runtime.TFLITE]: bertQaTfliteLoader,
    },
    CustomModel: {
      [Runtime.TFLITE]: qaCustomModelTfliteLoader,
    }
  },
};

// Export each task individually.

// tslint:disable:variable-name
export const ImageClassification = modelIndex[Task.IMAGE_CLASSIFICATION];
// tslint:disable:variable-name
export const ObjectDetection = modelIndex[Task.OBJECT_DETECTION];
// tslint:disable:variable-name
export const ImageSegmentation = modelIndex[Task.IMAGE_SEGMENTATION];
// tslint:disable:variable-name
export const SentimentDetection = modelIndex[Task.SENTIMENT_DETECTION];
// tslint:disable:variable-name
export const NLClassification = modelIndex[Task.NL_CLASSIFICATION];
// tslint:disable:variable-name
export const QuestionAndAnswer = modelIndex[Task.QUESTION_AND_ANSWER];

/**
 * Filter model loaders by runtimes.
 *
 * A model loader will be returned if its runtime matches any runtime in the
 * given array.
 */
export function getModelLoadersByRuntime(runtimes: Runtime[]):
    Array<TaskModelLoader<{}, {}, {}>> {
  return filterModelLoaders(
      (loader) => runtimes.includes(loader.metadata.runtime));
}

/** Gets all model loaders from the index. */
export function getAllModelLoaders(): Array<TaskModelLoader<{}, {}, {}>> {
  const modelLoaders: Array<TaskModelLoader<{}, {}, {}>> = [];
  for (const task of Object.keys(modelIndex)) {
    // tslint:disable-next-line:no-any
    const models = (modelIndex as any)[task];
    for (const modelName of Object.keys(models)) {
      const runtimes = models[modelName];
      for (const runtime of Object.keys(runtimes) as [keyof typeof runtimes]) {
        const taskModel = runtimes[runtime];
        modelLoaders.push(taskModel);
      }
    }
  }
  return modelLoaders;
}

function filterModelLoaders(
    filterFn: (loader: TaskModelLoader<{}, {}, {}>) => boolean) {
  const filteredModelLoaders: Array<TaskModelLoader<{}, {}, {}>> = [];
  for (const modelLoader of getAllModelLoaders()) {
    if (filterFn(modelLoader)) {
      filteredModelLoaders.push(modelLoader);
    }
  }
  return filteredModelLoaders;
}

// TODO: add more util functions as needed.
