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

// tslint:disable:max-line-length
import {BrowserFftFeatureExtractor, SpectrogramCallback} from './browser_fft_extractor';
import {loadMetadataJson} from './browser_fft_utils';
import {RecognizerCallback, RecognizerParams, SpectrogramData, SpeechCommandRecognizer, SpeechCommandRecognizerResult, StreamingRecognitionConfig, TransferLearnConfig, TransferSpeechCommandRecognizer} from './types';
import {version} from './version';

// tslint:enable:max-line-length

export const BACKGROUND_NOISE_TAG = '_background_noise_';
export const UNKNOWN_TAG = '_unknown_';

let streaming = false;

/**
 * Speech-Command Recognizer using browser-native (WebAudio) spectral featutres.
 */
export class BrowserFftSpeechCommandRecognizer implements
    SpeechCommandRecognizer {
  static readonly VALID_VOCABULARY_NAMES: string[] = ['18w', 'directional4w'];
  static readonly DEFAULT_VOCABULARY_NAME = '18w';

  readonly MODEL_URL_PREFIX =
      `https://storage.googleapis.com/tfjs-speech-commands-models/v${
          version}/browser_fft`;

  private readonly SAMPLE_RATE_HZ = 44100;
  private readonly FFT_SIZE = 1024;
  private readonly DEFAULT_SUPPRESSION_TIME_MILLIS = 1000;

  model: tf.Model;
  readonly vocabulary: string;
  readonly parameters: RecognizerParams;
  protected words: string[];

  protected nonBatchInputShape: [number, number, number];
  private elementsPerExample: number;
  protected audioDataExtractor: BrowserFftFeatureExtractor;

  private transferRecognizers:
      {[name: string]: TransferBrowserFftSpeechCommandRecognizer} = {};

  /**
   * Constructor of BrowserFftSpeechCommandRecognizer.
   */
  constructor(vocabulary?: string) {
    if (vocabulary == null) {
      vocabulary = BrowserFftSpeechCommandRecognizer.DEFAULT_VOCABULARY_NAME;
    }
    tf.util.assert(
        BrowserFftSpeechCommandRecognizer.VALID_VOCABULARY_NAMES.indexOf(
            vocabulary) !== -1,
        `Invalid vocabulary name: '${vocabulary}'`);
    this.vocabulary = vocabulary;
    this.parameters = {
      sampleRateHz: this.SAMPLE_RATE_HZ,
      fftSize: this.FFT_SIZE,
      columnBufferLength: this.FFT_SIZE,
    };
  }

  /**
   * Start streaming recognition.
   *
   * To stop the recognition, use `stopStreaming()`.
   *
   * Example: TODO(cais): Add exapmle code snippet.
   *
   * @param callback The callback invoked whenever a word is recognized
   *   with a probability score greater than `config.probabilityThreshold`.
   *   It has the signature:
   *     (result: SpeechCommandRecognizerResult) => Promise<void>
   *   wherein result has the two fields:
   *   - scores: A Float32Array that contains the probability scores for all
   *     the words.
   *   - spectrogram: The spectrogram data, provided only if
   *     `config.includeSpectrogram` is `true`.
   * @param config The configurations for the streaming recognition to
   *   be started.
   *   The `modelName` field of `config` specifies the model to be used for
   *   online recognition. If not specified, it defaults to the name of the
   *   base model ('base'), i.e., the pretrained model not from transfer
   *   learning. If the recognizer instance has one or more transfer-learning
   *   models ready (as a result of calls to `collectTransferExample`
   *   and `trainTransferModel`), you can let this call use that
   *   model for prediction by specifying the corresponding `modelName`.
   * @throws Error, if streaming recognition is already started or
   *   if `config` contains invalid values.
   */
  async startStreaming(
      callback: RecognizerCallback,
      config?: StreamingRecognitionConfig): Promise<void> {
    if (streaming) {
      throw new Error(
          'Cannot start streaming again when streaming is ongoing.');
    }

    await this.ensureModelLoaded();

    if (config == null) {
      config = {};
    }
    const probabilityThreshold =
        config.probabilityThreshold == null ? 0 : config.probabilityThreshold;
    tf.util.assert(
        probabilityThreshold >= 0 && probabilityThreshold <= 1,
        `Invalid probabilityThreshold value: ${probabilityThreshold}`);
    const invokeCallbackOnNoiseAndUnknown =
        config.invokeCallbackOnNoiseAndUnknown == null ?
        false :
        config.invokeCallbackOnNoiseAndUnknown;

    if (config.suppressionTimeMillis < 0) {
      throw new Error(
          `suppressionTimeMillis is expected to be >= 0, ` +
          `but got ${config.suppressionTimeMillis}`);
    }

    const overlapFactor =
        config.overlapFactor == null ? 0.5 : config.overlapFactor;
    tf.util.assert(
        overlapFactor >= 0 && overlapFactor < 1,
        `Expected overlapFactor to be >= 0 and < 1, but got ${overlapFactor}`);
    this.parameters.columnHopLength =
        Math.round(this.FFT_SIZE * (1 - overlapFactor));

    const spectrogramCallback: SpectrogramCallback = async (x: tf.Tensor) => {
      const y = tf.tidy(() => this.model.predict(x) as tf.Tensor);
      const scores = await y.data() as Float32Array;
      const maxIndexTensor = y.argMax(-1);
      const maxIndex = (await maxIndexTensor.data())[0];
      const maxScore = Math.max(...scores);
      tf.dispose([y, maxIndexTensor]);

      if (maxScore < probabilityThreshold) {
        return false;
      } else {
        let spectrogram: SpectrogramData = undefined;
        if (config.includeSpectrogram) {
          spectrogram = {
            data: await x.data() as Float32Array,
            frameSize: this.nonBatchInputShape[1],
          };
        }

        let wordDetected = true;
        if (!invokeCallbackOnNoiseAndUnknown) {
          // Skip background noise and unknown tokens.
          if (this.words[maxIndex] === BACKGROUND_NOISE_TAG ||
              this.words[maxIndex] === UNKNOWN_TAG) {
            wordDetected = false;
          }
        }
        if (wordDetected) {
          callback({scores, spectrogram});
        }
        // Trigger suppression only if the word is neither unknown or
        // background noise.
        return wordDetected;
      }
    };

    const suppressionTimeMillis = config.suppressionTimeMillis == null ?
        this.DEFAULT_SUPPRESSION_TIME_MILLIS :
        config.suppressionTimeMillis;
    this.audioDataExtractor = new BrowserFftFeatureExtractor({
      sampleRateHz: this.parameters.sampleRateHz,
      columnBufferLength: this.parameters.columnBufferLength,
      columnHopLength: this.parameters.columnHopLength,
      numFramesPerSpectrogram: this.nonBatchInputShape[0],
      columnTruncateLength: this.nonBatchInputShape[1],
      suppressionTimeMillis,
      spectrogramCallback
    });

    await this.audioDataExtractor.start();

    streaming = true;
  }

  /**
   * Load the underlying tf.Model instance and associated metadata.
   *
   * If the model and the metadata are already loaded, do nothing.
   */
  async ensureModelLoaded() {
    if (this.model != null) {
      return;
    }

    await this.ensureMetadataLoaded();

    const model = await tf.loadModel(
        `${this.MODEL_URL_PREFIX}/${this.vocabulary}/model.json`);
    // Check the validity of the model's input shape.
    if (model.inputs.length !== 1) {
      throw new Error(
          `Expected model to have 1 input, but got a model with ` +
          `${model.inputs.length} inputs`);
    }
    if (model.inputs[0].shape.length !== 4) {
      throw new Error(
          `Expected model to have an input shape of rank 4, ` +
          `but got an input shape of rank ${model.inputs[0].shape.length}`);
    }
    if (model.inputs[0].shape[3] !== 1) {
      throw new Error(
          `Expected model to have an input shape with 1 as the last ` +
          `dimension, but got input shape` +
          `${JSON.stringify(model.inputs[0].shape[3])}}`);
    }
    // Check the consistency between the word labels and the model's output
    // shape.
    const outputShape = model.outputShape as tf.Shape;
    if (outputShape.length !== 2) {
      throw new Error(
          `Expected loaded model to have an output shape of rank 2,` +
          `but received shape ${JSON.stringify(outputShape)}`);
    }
    if (outputShape[1] !== this.words.length) {
      throw new Error(
          `Mismatch between the last dimension of model's output shape ` +
          `(${outputShape[1]}) and number of words ` +
          `(${this.words.length}).`);
    }

    this.model = model;
    this.freezeModel();

    this.nonBatchInputShape =
        model.inputs[0].shape.slice(1) as [number, number, number];
    this.elementsPerExample = 1;
    model.inputs[0].shape.slice(1).forEach(
        dimSize => this.elementsPerExample *= dimSize);

    this.warmUpModel();

    const frameDurationMillis =
        this.parameters.columnBufferLength / this.parameters.sampleRateHz * 1e3;
    const numFrames = model.inputs[0].shape[1];
    this.parameters.spectrogramDurationMillis = numFrames * frameDurationMillis;
  }

  private warmUpModel() {
    tf.tidy(() => {
      const x = tf.zeros([1].concat(this.nonBatchInputShape));
      for (let i = 0; i < 3; ++i) {
        this.model.predict(x);
      }
    });
  }

  private async ensureMetadataLoaded() {
    if (this.words != null) {
      return;
    }
    const metadataJSON = await loadMetadataJson(
        `${this.MODEL_URL_PREFIX}/${this.vocabulary}/metadata.json`);
    this.words = metadataJSON.words;
  }

  /**
   * Stop streaming recognition.
   *
   * @throws Error if there is not ongoing streaming recognition.
   */
  async stopStreaming(): Promise<void> {
    if (!streaming) {
      throw new Error('Cannot stop streaming when streaming is not ongoing.');
    }
    await this.audioDataExtractor.stop();
    streaming = false;
  }

  /**
   * Check if streaming recognition is ongoing.
   */
  isStreaming(): boolean {
    return streaming;
  }

  /**
   * Get the array of word labels.
   *
   * @throws Error If this model is called before the model is loaded.
   */
  wordLabels(): string[] {
    return this.words;
  }

  /**
   * Get the parameters of this instance of BrowserFftSpeechCommandRecognizer.
   *
   * @returns Parameters of this instance.
   */
  params(): RecognizerParams {
    return this.parameters;
  }

  /**
   * Get the input shape of the underlying tf.Model.
   *
   * @returns The input shape.
   */
  modelInputShape(): tf.Shape {
    if (this.model == null) {
      throw new Error(
          'Model has not been loaded yet. Load model by calling ' +
          'ensureModelLoaded(), recognizer(), or startStreaming().');
    }
    return this.model.inputs[0].shape;
  }

  /**
   * Run offline (non-streaming) recognition on a spectrogram.
   *
   * @param input Spectrogram. Either a `tf.Tensor` of a `Float32Array`.
   *   - If a `tf.Tensor`, must be rank-4 and match the model's expected
   *     input shape in 2nd dimension (# of spectrogram columns), the 3rd
   *     dimension (# of frequency-domain points per column), and the 4th
   *     dimension (always 1). The 1st dimension can be 1, for single-example
   *     recogntion, or any value >1, for batched recognition.
   *   - If a `Float32Array`, must have a length divisible by the number
   *     of elements per spectrogram, i.e.,
   *     (# of spectrogram columns) * (# of frequency-domain points per column).
   * @returns Result of the recognition, with the following field:
   *   scores:
   *   - A `Float32Array` if there is only one input exapmle.
   *   - An `Array` of `Float32Array`, if there are multiple input examples.
   */
  async recognize(input: tf.Tensor|
                  Float32Array): Promise<SpeechCommandRecognizerResult> {
    await this.ensureModelLoaded();

    let numExamples: number;
    let inputTensor: tf.Tensor;
    let outTensor: tf.Tensor;
    if (input instanceof tf.Tensor) {
      // Check input shape.
      this.checkInputTensorShape(input);
      inputTensor = input;
      numExamples = input.shape[0];
    } else {
      // `input` is a `Float32Array`.
      input = input as Float32Array;
      if (input.length % this.elementsPerExample) {
        throw new Error(
            `The length of the input Float32Array ${input.length} ` +
            `is not divisible by the number of tensor elements per ` +
            `per example expected by the model ${this.elementsPerExample}.`);
      }

      numExamples = input.length / this.elementsPerExample;
      inputTensor = tf.tensor4d(input, [
        numExamples
      ].concat(this.nonBatchInputShape) as [number, number, number, number]);
    }

    outTensor = this.model.predict(inputTensor) as tf.Tensor;
    if (numExamples === 1) {
      return {scores: await outTensor.data() as Float32Array};
    } else {
      const unstacked = tf.unstack(outTensor) as tf.Tensor[];
      const scorePromises = unstacked.map(item => item.data());
      const scores = await Promise.all(scorePromises) as Float32Array[];
      tf.dispose(unstacked);
      return {scores};
    }
  }

  createTransfer(name: string): TransferSpeechCommandRecognizer {
    if (this.model == null) {
      throw new Error(
          'Model has not been loaded yet. Load model by calling ' +
          'ensureModelLoaded(), recognizer(), or startStreaming().');
    }
    tf.util.assert(
        name != null && typeof name === 'string' && name.length > 1,
        `Expected the name for a transfer-learning recognized to be a ` +
            `non-empty string, but got ${JSON.stringify(name)}`);
    tf.util.assert(
        this.transferRecognizers[name] == null,
        `There is already a transfer-learning model named '${name}'`);
    const transfer = new TransferBrowserFftSpeechCommandRecognizer(
        name, this.parameters, this.model);
    this.transferRecognizers[name] = transfer;
    return transfer;
  }

  private freezeModel(): void {
    for (const layer of this.model.layers) {
      layer.trainable = false;
    }
  }

  private checkInputTensorShape(input: tf.Tensor) {
    const expectedRank = this.model.inputs[0].shape.length;
    if (input.shape.length !== expectedRank) {
      throw new Error(
          `Expected input Tensor to have rank ${expectedRank}, ` +
          `but got rank ${input.shape.length} that differs `);
    }
    const nonBatchedShape = input.shape.slice(1);
    const expectedNonBatchShape = this.model.inputs[0].shape.slice(1);
    if (!tf.util.arraysEqual(nonBatchedShape, expectedNonBatchShape)) {
      throw new Error(
          `Expected input to have shape [null,${expectedNonBatchShape}], ` +
          `but got shape [null,${nonBatchedShape}]`);
    }
  }

  // TODO(cais): Implement model save and load.
}

/**
 * A subclass of BrowserFftSpeechCommandRecognizer: Transfer-learned model.
 */
class TransferBrowserFftSpeechCommandRecognizer extends
    BrowserFftSpeechCommandRecognizer implements
        TransferSpeechCommandRecognizer {
  private transferExamples: {[word: string]: tf.Tensor[]};
  private transferHead: tf.Sequential;

  /**
   * Constructor of TransferBrowserFftSpeechCommandRecognizer.
   *
   * @param name Name of the transfer-learned recognizer. Must be a non-empty
   *   string.
   * @param parameters Parameters from the base recognizer.
   * @param baseModel Model from the base recognizer.
   */
  constructor(
      readonly name: string, readonly parameters: RecognizerParams,
      readonly baseModel: tf.Model) {
    super();
    tf.util.assert(
        name != null && typeof name === 'string' && name.length > 0,
        `The name of a transfer model must be a non-empty string, ` +
            `but got ${JSON.stringify(name)}`);
    this.nonBatchInputShape =
        this.baseModel.inputs[0].shape.slice(1) as [number, number, number];
    this.words = [];
  }

  /**
   * Collect an example for transfer learning via WebAudio.
   *
   * @param {string} word Name of the word. Must not overlap with any of the
   *   words the base model is trained to recognize.
   * @returns {SpectrogramData} The spectrogram of the acquired the example.
   * @throws Error, if word belongs to the set of words the base model is
   *   trained to recognize.
   */
  async collectExample(word: string): Promise<SpectrogramData> {
    tf.util.assert(
        !streaming,
        'Cannot start collection of transfer-learning example because ' +
            'a streaming recognition or transfer-learning example collection ' +
            'is ongoing');
    tf.util.assert(
        word != null && typeof word === 'string' && word.length > 0,
        `Must provide a non-empty string when collecting transfer-` +
            `learning example`);

    streaming = true;
    return new Promise<SpectrogramData>((resolve, reject) => {
      const spectrogramCallback: SpectrogramCallback = async (x: tf.Tensor) => {
        if (this.transferExamples == null) {
          this.transferExamples = {};
        }
        if (this.transferExamples[word] == null) {
          this.transferExamples[word] = [];
        }
        this.transferExamples[word].push(x.clone());
        await this.audioDataExtractor.stop();
        streaming = false;
        this.collateTransferWords();
        resolve({
          data: await x.data() as Float32Array,
          frameSize: this.nonBatchInputShape[1],
        });
        return false;
      };
      this.audioDataExtractor = new BrowserFftFeatureExtractor({
        sampleRateHz: this.parameters.sampleRateHz,
        columnBufferLength: this.parameters.columnBufferLength,
        columnHopLength: this.parameters.columnBufferLength,
        numFramesPerSpectrogram: this.nonBatchInputShape[0],
        columnTruncateLength: this.nonBatchInputShape[1],
        suppressionTimeMillis: 0,
        spectrogramCallback
      });
      this.audioDataExtractor.start();
    });
  }

  /**
   * Clear all transfer learning examples collected so far.
   */
  clearExamples(): void {
    tf.util.assert(
        this.words != null && this.words.length > 0 &&
            this.transferExamples != null,
        `No transfer learning examples exist for model name ${this.name}`);
    tf.dispose(this.transferExamples);
    this.transferExamples = null;
    this.words = null;
  }

  /**
   * Get counts of the word examples that have been collected for a
   * transfer-learning model.
   *
   * @returns {{[word: string]: number}} A map from word name to number of
   *   examples collected for that word so far.
   */
  countExamples(): {[word: string]: number} {
    if (this.transferExamples == null) {
      throw new Error(
          `No examples have been collected for transfer-learning model ` +
          `named '${this.name}' yet.`);
    }
    const counts: {[word: string]: number} = {};
    for (const word in this.transferExamples) {
      counts[word] = this.transferExamples[word].length;
    }
    return counts;
  }

  /**
   * Collect the vocabulary of this transfer-learned recognizer.
   *
   * The words are put in an alphabetically sorted order.
   */
  private collateTransferWords() {
    this.words = Object.keys(this.transferExamples).sort();
  }

  /**
   * Collect the transfer-learning data as tf.Tensors.
   *
   * @param modelName {string} Name of the transfer learning model for which
   *   the examples are to be collected.
   * @returns xs: The feature tensors (xs), a 4D tf.Tensor.
   *          ys: The target tensors (ys), one-hot encoding, a 2D tf.Tensor.
   */
  private collectTransferDataAsTensors(modelName?: string):
      {xs: tf.Tensor, ys: tf.Tensor} {
    tf.util.assert(
        this.words != null && this.words.length > 0,
        `No word example is available for tranfer-learning model of name ` +
            modelName);
    return tf.tidy(() => {
      const xTensors: tf.Tensor[] = [];
      const targetIndices: number[] = [];
      this.words.forEach((word, i) => {
        this.transferExamples[word].forEach(wordTensor => {
          xTensors.push(wordTensor);
          targetIndices.push(i);
        });
      });
      return {
        xs: tf.concat(xTensors, 0),
        ys: tf.oneHot(
            tf.tensor1d(targetIndices, 'int32'), Object.keys(this.words).length)
      };
    });
  }

  /**
   * Train the transfer-learning model.
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
  async train(config?: TransferLearnConfig): Promise<tf.History> {
    tf.util.assert(
        this.words != null && this.words.length > 0,
        `Cannot train transfer-learning model '${this.name}' because no ` +
            `transfer learning example has been collected.`);
    tf.util.assert(
        this.words.length > 1,
        `Cannot train transfer-learning model '${this.name}' because only ` +
            `1 word label ('${JSON.stringify(this.words)}') ` +
            `has been collected for transfer learning. Requires at least 2.`);

    if (config == null) {
      config = {};
    }

    if (this.model == null) {
      this.createTransferModelFromBaseModel();
    }

    // Compile model for training.
    const optimizer = config.optimizer || 'sgd';
    this.model.compile(
        {loss: 'categoricalCrossentropy', optimizer, metrics: ['acc']});

    // Prepare the data.
    const {xs, ys} = this.collectTransferDataAsTensors();

    const epochs = config.epochs == null ? 20 : config.epochs;
    const validationSplit =
        config.validationSplit == null ? 0 : config.validationSplit;
    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        validationSplit,
        batchSize: config.batchSize,
        callbacks: config.callback == null ? null : [config.callback]
      });
      tf.dispose([xs, ys]);
      return history;
    } catch (err) {
      tf.dispose([xs, ys]);
      this.model = null;
      return null;
    }
  }

  /**
   * Create an instance of tf.Model for transfer learning.
   *
   * The top dense layer of the base model is replaced with a new softmax
   * dense layer.
   */
  private createTransferModelFromBaseModel(): void {
    tf.util.assert(
        this.words != null,
        `No word example is available for tranfer-learning model of name ` +
            this.name);

    // Find the second last dense layer.
    const layers = this.baseModel.layers;
    let layerIndex = layers.length - 2;
    while (layerIndex >= 0) {
      if (layers[layerIndex].getClassName().toLowerCase() === 'dense') {
        break;
      }
      layerIndex--;
    }
    if (layerIndex < 0) {
      throw new Error('Cannot find a hidden dense layer in the base model.');
    }
    const beheadedBaseOutput = layers[layerIndex].output as tf.SymbolicTensor;

    this.transferHead = tf.sequential();
    this.transferHead.add(tf.layers.dense({
      units: this.words.length,
      activation: 'softmax',
      inputShape: beheadedBaseOutput.shape.slice(1)
    }));
    const transferOutput =
        this.transferHead.apply(beheadedBaseOutput) as tf.SymbolicTensor;
    this.model =
        tf.model({inputs: this.baseModel.inputs, outputs: transferOutput});
  }

  /**
   * Get the input shape of the underlying tf.Model.
   *
   * @returns The input shape.
   */
  modelInputShape(): tf.Shape {
    return this.baseModel.inputs[0].shape;
  }

  /**
   * Overridden method to prevent creating a nested transfer-learning
   * recognizer.
   *
   * @param name
   */
  createTransfer(name: string): TransferBrowserFftSpeechCommandRecognizer {
    throw new Error(
        'Creating transfer-learned recognizer from a transfer-learned ' +
        'recognizer is not supported.');
  }
}