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
import {InferenceModel} from '@tensorflow/tfjs';

export interface Params {
  inputBufferLength?: number;
  bufferLength?: number;
  hopLength?: number;
  duration?: number;
  fftSize?: number;
  melCount?: number;
  targetSr?: number;
  isMfccEnabled?: boolean;
}

export interface FeatureExtractor {
  config(params: Params): void;
  start(samples?: Float32Array): Promise<Float32Array[]>|void;
  stop(): void;
  getFeatures(): Float32Array[];
  getImages(): Float32Array[];
}

export enum ModelType {
  FROZEN_MODEL = 0,
  FROZEN_MODEL_NATIVE,
  TF_MODEL
}

export const BUFFER_LENGTH = 1024;
export const HOP_LENGTH = 444;
export const MEL_COUNT = 40;
export const EXAMPLE_SR = 44100;
export const DURATION = 1.0;
export const IS_MFCC_ENABLED = true;
export const MIN_SAMPLE = 3;
export const DETECTION_THRESHOLD = 0.5;
export const SUPPRESSION_TIME = 500;
export const MODELS: {[key: number]: InferenceModel} = {};
