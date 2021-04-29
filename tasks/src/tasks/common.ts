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

/** Default TFJS backend. */
export const DEFAULT_TFJS_BACKEND: TFJSBackend = 'webgl';

/** Default TFJS version. */
export const DEFAULT_TFJS_VERSION = '3.5.0';

/** Type of TFJS bckends. */
export type TFJSBackend = 'cpu'|'webgl'|'wasm';

/** Common loading options for TFJS models. */
export interface TFJSModelCommonLoadingOption {
  backend: TFJSBackend;
}

/** Common loading goptions for TFLite models. */
export interface TFLiteCustomModelCommonLoadingOption {
  model: string|ArrayBuffer;
}

/** All supported tasks. */
export enum Task {
  IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION',
}

/** All supported runtimes. */
export enum Runtime {
  TFJS = 'TFJS',
  TFLITE = 'TFLite',
  MEDIA_PIPE = 'MediaPipe',
}

/** A helper function to get the TFJS packages that a TFJS model depends on. */
export function getTFJSModelDependencyPackages(
    backend = DEFAULT_TFJS_BACKEND,
    version = DEFAULT_TFJS_VERSION): Array<string[]> {
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
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@${
              version}`);
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@${
              version}`);
      break;
    case 'wasm':
      packages[1].push(
          `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
              version}/dist/tf-backend-wasm.js`);
      break;
  }
  return packages;
}
