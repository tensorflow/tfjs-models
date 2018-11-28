/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';

/**
 * This file defines the interfaces related to SpeechCommandRecognizer.
 */

export type FFT_TYPE = 'BROWSER_FFT'|'SOFT_FFT';

export type RecognizerCallback = (result: SpeechCommandRecognizerResult) =>
    Promise<void>;

/**
 * Interface for a speech-command recognizer.
 */
export interface SpeechCommandRecognizer {
  /**
   * Load the underlying model instance and associated metadata.
   *
   * If the model and the metadata are already loaded, do nothing.
   */
  ensureModelLoaded(): Promise<void>;

  /**
   * Start listening continuously to microphone input and perform recognition
   * in a streaming fashion.
   *
   * @param callback the callback that will be invoked every time
   *   a recognition result is available.
   * @param config optional configuration.
   * @throws Error if there is already ongoing streaming recognition.
   */
  listen(callback: RecognizerCallback, config?: StreamingRecognitionConfig):
      Promise<void>;

  /**
   *  Stop the ongoing streaming recognition (if any).
   *
   * @throws Error if no streaming recognition is ongoing.
   */
  stopListening(): Promise<void>;

  /**
   * Check if this instance is currently performing
   * streaming recognition.
   */
  isListening(): boolean;

  /**
   * Recognize a single example of audio.
   *
   * If `input` is provided, will perform offline prediction.
   * If `input` is not provided, a single frame of audio
   *   will be collected from the microhpone via WebAudio and predictions
   *   will be made on it.
   *
   * @param input (Optional) tf.Tensor of Float32Array.
   *     If provided and a tf.Tensor, must match the input shape of the
   *     underlying tf.Model. If a Float32Array, the length must be
   *     equal to (the model’s required FFT length) *
   *     (the model’s required frame count).
   * @returns A Promise of recognition result, with the following fields:
   *   - scores: the probability scores.
   *   - embedding: the embedding for the input audio (i.e., an internal
   *     activation from the model). Provided if and only if `includeEmbedding`
   *     is `true` in `config`.
   * @throws Error on incorrect shape or length.
   */
  recognize(input?: tf.Tensor|Float32Array, config?: RecognizeConfig):
      Promise<SpeechCommandRecognizerResult>;

  /**
   * Get the input shape of the tf.Model the underlies the recognizer.
   */
  modelInputShape(): tf.Shape;

  /**
   * Getter for word labels.
   *
   * The word labels are an alphabetically sorted Array of strings.
   */
  wordLabels(): string[];

  /**
   * Get the parameters such as the required number of frames.
   */
  params(): RecognizerParams;

  /**
   * Create a new recognizer based on this recognizer, for transfer learning.
   *
   * @param name Required name of the transfer learning recognizer. Must be a
   *   non-empty string.
   * @returns An instance of TransferSpeechCommandRecognizer, which supports
   *     `collectExample()`, `train()`, as well as the same `listen()`
   *     `stopListening()` and `recognize()` as the base recognizer.
   */
  createTransfer(name: string): TransferSpeechCommandRecognizer;
}

/**
 * Interface for a transfer-learning speech command recognizer.
 *
 * This inherits the `SpeechCommandRecognizer`. It adds methods for
 * collecting and clearing examples for transfer learning, methods for
 * querying the status of example collection, and for performing the
 * transfer-learning training.
 */
export interface TransferSpeechCommandRecognizer extends
    SpeechCommandRecognizer {
  /**
   * Collect an example for transfer learning via WebAudio.
   *
   * @param {string} word Name of the word. Must not overlap with any of the
   *   words the base model is trained to recognize.
   * @returns {SpectrogramData} The spectrogram of the acquired the example.
   * @throws Error, if word belongs to the set of words the base model is
   *   trained to recognize.
   */
  collectExample(word: string): Promise<SpectrogramData>;

  /**
   * Clear all transfer learning examples collected so far.
   */
  clearExamples(): void;

  /**
   * Get counts of the word examples that have been collected for a
   * transfer-learning model.
   *
   * @returns {{[word: string]: number}} A map from word name to number of
   *   examples collected for that word so far.
   */
  countExamples(): {[word: string]: number};

  /**
   * Train a transfer-learning model.
   *
   * The last dense layer of the base model is replaced with new softmax dense
   * layer.
   *
   * It is assume that at least one category of data has been collected (using
   * multiple calls to the `collectTransferExample` method).
   *
   * @param config {TransferLearnConfig} Optional configurations fot the
   *   training of the transfer-learning model.
   * @returns {tf.History} A history object with the loss and accuracy values
   *   from the training of the transfer-learning model.
   * @throws Error, if `modelName` is invalid or if not sufficient training
   *   examples have been collected yet.
   */
  train(config?: TransferLearnConfig):
      Promise<tf.History|[tf.History, tf.History]>;

  /**
   * Get examples currently held by the transfer-learning recognizer.
   *
   * @param label Label requested.
   * @returns An array of `Example`s, along with their UIDs.
   */
  getExamples(label: string): Array<{uid: string, example: Example}>;

  /**
   * Load an array of serialized examples.
   *
   * @param serialized The examples in their serialized format.
   * @param clearExisting Whether to clear the existing examples while
   *   performing the loading (default: false).
   */
  loadExamples(serialized: ArrayBuffer, clearExisting?: boolean): void;

  /** Serialize the existing examples. */
  serializeExamples(): ArrayBuffer;
}

/**
 * Interface for a snippet of audio spectrogram.
 */
export interface SpectrogramData {
  /**
   * The float32 data for the spectrogram.
   *
   * Stored frame by frame. For example, the first N elements
   * belong to the first time frame and the next N elements belong
   * to the second time frame, and so forth.
   */
  data: Float32Array;

  /**
   * Number of points per frame, i.e., FFT length per frame.
   */
  frameSize: number;
}

/**
 * Interface for a result emitted by a speech-command recognizer.
 *
 * It is used in the callback of a recognizer's streaming or offline
 * recognition method. It represents the result for a short snippet of
 * audio.
 */
export interface SpeechCommandRecognizerResult {
  /**
   * Probability scores for the words.
   */
  scores: Float32Array|Float32Array[];

  /**
   * Optional spectrogram data.
   */
  spectrogram?: SpectrogramData;

  /**
   * Embedding (internal activation) for the input.
   *
   * This field is populated if and only if `includeEmbedding`
   * is `true` in the configuration object used during the `recognize` call.
   */
  embedding?: tf.Tensor;
}

export interface StreamingRecognitionConfig {
  /**
   * Overlap factor. Must be >=0 and <1.
   * Defaults to 0.5.
   * For example, if the model takes a frame length of 1000 ms,
   * and if overlap factor is 0.4, there will be a 400ms
   * overlap between two successive frames, i.e., frames
   * will be taken every 600 ms.
   */
  overlapFactor?: number;

  /**
   * Amount to time in ms to suppress recognizer after a word is recognized.
   *
   * Defaults to 1000 ms.
   */
  suppressionTimeMillis?: number;

  /**
   * Threshold for the maximum probability value in a model prediction
   * output to be greater than or equal to, below which the callback
   * will not be called.
   *
   * Must be a number >=0 and <=1.
   *
   * The value will be overridden to `0` if `includeEmbedding` is `true`.
   *
   * If `null` or `undefined`, will default to `0`.
   */
  probabilityThreshold?: number;

  /**
   * Invoke the callback for background noise and unknown.
   *
   * The value will be overridden to `true` if `includeEmbedding` is `true`.
   *
   * Default: `false`.
   */
  invokeCallbackOnNoiseAndUnknown?: boolean;

  /**
   * Whether the spectrogram is to be provided in the each recognition
   * callback call.
   *
   * Default: `false`.
   */
  includeSpectrogram?: boolean;

  /**
   * Whether to include the embedding (internal activation).
   *
   * If set as `true`, the values of the following configuration fields
   * in this object will be overridden:
   *
   * - `probabilityThreshold` will be overridden to 0.
   * - `invokeCallbackOnNoiseAndUnknown` will be overridden to `true`.
   *
   * Default: `false`.
   */
  includeEmbedding?: boolean;
}

export interface RecognizeConfig {
  /**
   * Whether the spectrogram is to be provided in the each recognition
   * callback call.
   *
   * Default: `false`.
   */
  includeSpectrogram?: boolean;

  /**
   * Whether to include the embedding (internal activation).
   *
   * Default: `false`.
   */
  includeEmbedding?: boolean;
}

/**
 * Configurations for the training of a transfer-learning recognizer.
 *
 * It is used during calls to the `TransferSpeechCommandRecognizer.train()`
 * method.
 */
export interface TransferLearnConfig {
  /**
   * Number of training epochs (default: 20).
   */
  epochs?: number;

  /**
   * Optimizer to be used for training (default: 'sgd').
   */
  optimizer?: string|tf.Optimizer;

  /**
   * Batch size of training (default: 128).
   */
  batchSize?: number;

  /**
   * Validation split to be used during training.
   *
   * Default: null (no validation split).
   *
   * Note that this is split is different from the basic validation-split
   * paradigm in TensorFlow.js. It makes sure that the distribution of the
   * classes in the training and validation sets are approximately balanced.
   *
   * If specified, must be a number > 0 and < 1.
   */
  validationSplit?: number;

  /**
   * Number of fine-tuning epochs to run after the initial `epochs` epochs
   * of transfer-learning training.
   *
   * During the fine-tuning, the last dense layer of the truncated base
   * model (i.e., the second-last dense layer of the original model) is
   * unfrozen and updated through backpropagation.
   *
   * If specified, must be an integer > 0.
   */
  fineTuningEpochs?: number;

  /**
   * The optimizer for fine-tuning after the initial transfer-learning
   * training.
   *
   * This parameter is used only if `fineTuningEpochs` is specified
   * and is a positive integre.
   *
   * Default: 'sgd'.
   */
  fineTuningOptimizer?: string|tf.Optimizer;

  /**
   * tf.Callback to be used during the initial training (i.e., not
   * the fine-tuning phase).
   */
  callback?: tf.CustomCallbackConfig;

  /**
   * tf.Callback to be used durnig the fine-tuning phase.
   *
   * This parameter is used only if `fineTuningEpochs` is specified
   * and is a positive integer.
   */
  fineTuningCallback?: tf.CustomCallbackConfig;
}

/**
 * Parameters for a speech-command recognizer.
 */
export interface RecognizerParams {
  /**
   * Total duration per spectragram, in milliseconds.
   */
  spectrogramDurationMillis?: number;

  /**
   * FFT encoding size per spectrogram column.
   */
  fftSize?: number;

  /**
   * Sampling rate, in Hz.
   */
  sampleRateHz?: number;
}

/**
 * Interface of an audio feature extractor.
 */
export interface FeatureExtractor {
  /**
   * Config the feature extractor.
   */
  setConfig(params: RecognizerParams): void;

  /**
   * Start the feature extraction from the audio samples.
   */
  start(): Promise<Float32Array[]|void>;

  /**
   * Stop the feature extraction.
   */
  stop(): Promise<void>;

  /**
   * Get the extractor features collected since last call.
   */
  getFeatures(): Float32Array[];
}

/** Snippet of pulse-code modulation (PCM) audio data. */
export interface RawAudioData {
  /** Samples of the snippet. */
  data: Float32Array;

  /** Sampling rate, in Hz. */
  sampleRateHz: number;
}

/**
 * A short, labeled snippet of speech or audio.
 *
 * This can be used for training a transfer model based on the base
 * speech-commands model, among other things.
 *
 * A set of `Example`s can make up a dataset.
 */
export interface Example {
  /** A label for the example. */
  label: string;

  /** Spectrogram data. */
  spectrogram: SpectrogramData;

  /**
   * Raw audio in PCM (pulse-code modulation) format.
   *
   * Optional.
   */
  rawAudio?: RawAudioData;
}
