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

export interface FeatureExtractor extends EventEmitter.EventEmitter {
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
