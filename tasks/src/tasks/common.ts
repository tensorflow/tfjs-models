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

import {isWebWorker} from '../utils';

/** Default TFJS backend. */
export const DEFAULT_TFJS_BACKEND: TFJSBackend = 'webgl';

/** Default TFJS version. */
export const DEFAULT_TFJS_VERSION = '3.6.0';

/**
 * Type of TFJS bckends.
 *
 * @docinline
 */
export type TFJSBackend = 'cpu'|'webgl'|'wasm';

/** Common loading options for TFJS models. */
export interface TFJSModelCommonLoadingOption {
  /** The backend to use to run TFJS models. Default to 'webgl'. */
  backend: TFJSBackend;
}

/** Common loading options for custom TFLite models. */
export interface TFLiteCustomModelCommonLoadingOption {
  /**
   * The model url, or the model content stored in an `ArrayBuffer`.
   *
   * You can use TFLite model urls from `tfhub.dev` directly. For model
   * compatibility, see comments in the corresponding model class.
   */
  model: string|ArrayBuffer;
}

/** All supported tasks. */
export enum Task {
  IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION',
  OBJECT_DETECTION = 'OBJECT_DETECTION',
  IMAGE_SEGMENTATION = 'IMAGE_SEGMENTATION',
  SENTIMENT_DETECTION = 'SENTIMENT_DETECTION',
  NL_CLASSIFICATION = 'NL_CLASSIFICATION',
  QUESTION_AND_ANSWER = 'QUESTION_AND_ANSWER',
}

/** All supported runtimes. */
export enum Runtime {
  TFJS = 'TFJS',
  TFLITE = 'TFLite',
  MEDIA_PIPE = 'MediaPipe',
}

/** A single class in the classification result. */
export interface Class {
  /** The name of the class. */
  className: string;
  /** The score of the class. */
  score: number;
}

/** A helper function to get the TFJS packages that a TFJS model depends on. */
export function getTFJSModelDependencyPackages(
    backend = DEFAULT_TFJS_BACKEND,
    version = DEFAULT_TFJS_VERSION): string[][] {
  const packages = [
    [`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@${version}`],
    [`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@${version}`],
  ];
  switch (backend) {
    case 'cpu':
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@${
              version}`);
      break;
    case 'webgl':
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@${
              version}`);
      break;
    case 'wasm':
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
              version}/dist/tf-backend-wasm.min.js`);
      break;
    default:
      console.warn(
          'WARNING',
          `Backend '${backend}' not supported. Use 'webgl' as the default.`);
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@${
              version}`);
      break;
  }
  return packages;
}

/**
 * Makes sure the current tfjs backend matches the one in the given option.
 *
 * For TFJS models, this function should be called at the loading time as well
 * as before running inference.
 *
 * Users might run multiple TFJS models with different backend options in a web
 * app. Only setting the backend at the model loading time is not enough because
 * the backend might be set to another one when loading a different model. We
 * also need to call this right before running the inference.
 */
export async function ensureTFJSBackend(
    options?: TFJSModelCommonLoadingOption) {
  const backend: TFJSBackend = options ? options.backend : DEFAULT_TFJS_BACKEND;
  // tslint:disable-next-line:no-any
  const global: any = isWebWorker() ? self : window;
  const tf = global['tf'];
  if (!tf) {
    throw new Error('tfjs not loaded');
  }
  if (tf.getBackend() !== backend) {
    await tf.setBackend(backend);
  }
}
