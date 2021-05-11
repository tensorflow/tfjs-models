/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

/*
  Each pixel is assigned a label in the segmentation map that DeepLab outputs.
*/
export type Label = number;

/*
  Each label in the segmentation map has a corresponding color.
*/
export type Color = [number, number, number];

/*
  A dictionary interface for recognized classes and their colors
*/
export interface Legend {
  [name: string]: Color;
}

/**
 * The model supports quantization to 1 and 2 bytes, leaving 4 for the
 * non-quantized variant.
 *
 * @docinline
 */
export type QuantizationBytes = 1|2|4;
/**
 * Three types of pre-trained weights are available, trained on Pascal,
 * Cityscapes and ADE20K datasets. Each dataset has its own colormap and
 * labelling scheme.
 *
 * @docinline
 */
export type ModelArchitecture = 'pascal'|'cityscapes'|'ade20k';

export type DeepLabInput =
    |ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;

/*
 * The model can be configured with any of the following attributes:
 */
export interface ModelConfig {
  /**
   * The degree to which weights are quantized (either 1, 2 or 4).
   * Setting this attribute to 1 or 2 will load the model with int32 and
   * float32 compressed to 1 or 2 bytes respectively.
   * Set it to 4 to disable quantization.
   */
  quantizationBytes?: QuantizationBytes;
  /**
   * The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
   */
  base?: ModelArchitecture;
  /**
   *
   * The URL from which to load the TF.js GraphModel JSON.
   * Inferred from `base` and `quantizationBytes` if undefined.
   */
  modelUrl?: string;
}

/*
 * Segmentation can be fine-tuned with three parameters:
 */
export interface PredictionConfig {
  /** The canvas where to draw the output. */
  canvas?: HTMLCanvasElement;
  /** The array of RGB colors corresponding to labels. */
  colormap?: Color[];
  /**
   * The array of names corresponding to labels.
   *
   * By [default](./src/index.ts#L81), `colormap` and `labels` are set
   * according to the `base` model attribute passed during initialization.
   */
  labels?: string[];
}

/* The raw segmentation map, with labels assigned to each label, can be
 * can be converted to a human-readable format with two objects:
 *
 * - **legend** :: `{ [name: string]: [number, number, number] }`
 *
 *   The legend is a dictionary of objects recognized in the image and their
 *   colors in RGB format.
 *
 * - **segmentationMap** :: `Uint8ClampedArray`
 *
 *   The colored segmentation map as `Uint8ClampedArray` which can be
 *   fed into `ImageData` and mapped to a canvas.
 */
export interface SegmentationData {
  legend: Legend;
  segmentationMap: Uint8ClampedArray;
}

/* The model accepts an image, canvas or video, and outputs the promise of an
 * object with four attributes:
 *
 * - **legend** :: `{ [name: string]: [number, number, number] }`
 *
 *   The legend is a dictionary of objects recognized in the image and their
 *   colors in RGB format.
 *
 * - **height** :: `number`
 *
 *   The height of the returned segmentation map
 *
 * - **width** :: `number`
 *
 *   The width of the returned segmentation map
 *
 * - **segmentationMap** :: `Uint8ClampedArray`
 *
 *   The colored segmentation map as `Uint8ClampedArray` which can be
 *   fed into `ImageData` and mapped to a canvas.
 */
export interface DeepLabOutput {
  legend: Legend;
  height: number;
  width: number;
  segmentationMap: Uint8ClampedArray;
}
