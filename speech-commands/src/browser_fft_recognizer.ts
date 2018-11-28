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

import {BrowserFftFeatureExtractor, SpectrogramCallback} from './browser_fft_extractor';
import {loadMetadataJson, normalize} from './browser_fft_utils';
import {Dataset, SerializedExamples} from './dataset';
import {balancedTrainValSplit} from './training_utils';
import {Example, RecognizeConfig, RecognizerCallback, RecognizerParams, SpectrogramData, SpeechCommandRecognizer, SpeechCommandRecognizerResult, StreamingRecognitionConfig, TransferLearnConfig, TransferSpeechCommandRecognizer} from './types';
import {version} from './version';

export const BACKGROUND_NOISE_TAG = '_background_noise_';
export const UNKNOWN_TAG = '_unknown_';

let streaming = false;

export function getMajorAndMinorVersion(version: string) {
  const versionItems = version.split('.');
  return versionItems.slice(0, 2).join('.');
}

/**
 * Speech-Command Recognizer using browser-native (WebAudio) spectral featutres.
 */
export class BrowserFftSpeechCommandRecognizer implements
    SpeechCommandRecognizer {
  static readonly VALID_VOCABULARY_NAMES: string[] = ['18w', 'directional4w'];
  static readonly DEFAULT_VOCABULARY_NAME = '18w';

  readonly MODEL_URL_PREFIX =
      `https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v${
          getMajorAndMinorVersion(version)}/browser_fft`;

  private readonly SAMPLE_RATE_HZ = 44100;
  private readonly FFT_SIZE = 1024;
  private readonly DEFAULT_SUPPRESSION_TIME_MILLIS = 0;

  model: tf.Model;
  modelWithEmbeddingOutput: tf.Model;
  readonly vocabulary: string;
  readonly parameters: RecognizerParams;
  protected words: string[];

  protected nonBatchInputShape: [number, number, number];
  private elementsPerExample: number;
  protected audioDataExtractor: BrowserFftFeatureExtractor;

  private transferRecognizers:
      {[name: string]: TransferBrowserFftSpeechCommandRecognizer} = {};

  private modelURL: string;
  private metadataURL: string;

  // The second-last dense layer in the base model.
  // To be used for unfreezing during fine-tuning.
  protected secondLastBaseDenseLayer: tf.layers.Layer;

  /**
   * Constructor of BrowserFftSpeechCommandRecognizer.
   *
   * @param vocabulary An optional vocabulary specifier. Mutually exclusive
   *   with `modelURL` and `metadataURL`.
   * @param modelURL An optional, custom model URL pointing to a model.json
   *   file. Supported schemes: http://, https://, and node.js-only: file://.
   *   Mutually exclusive with `vocabulary`. If provided, `metadatURL`
   *   most also be provided.
   * @param metadataURL A custom metadata URL pointing to a metadata.json
   *   file. Must be provided together with `modelURL`.
   */
  constructor(vocabulary?: string, modelURL?: string, metadataURL?: string) {
    tf.util.assert(
        modelURL == null && metadataURL == null ||
            modelURL != null && metadataURL != null,
        `modelURL and metadataURL must be both provided or ` +
            `both not provided.`);
    if (modelURL == null) {
      if (vocabulary == null) {
        vocabulary = BrowserFftSpeechCommandRecognizer.DEFAULT_VOCABULARY_NAME;
      } else {
        tf.util.assert(
            BrowserFftSpeechCommandRecognizer.VALID_VOCABULARY_NAMES.indexOf(
                vocabulary) !== -1,
            `Invalid vocabulary name: '${vocabulary}'`);
      }
      this.vocabulary = vocabulary;
      this.modelURL = `${this.MODEL_URL_PREFIX}/${this.vocabulary}/model.json`;
      this.metadataURL =
          `${this.MODEL_URL_PREFIX}/${this.vocabulary}/metadata.json`;
    } else {
      tf.util.assert(
          vocabulary == null,
          `vocabulary name must be null or undefined when modelURL is ` +
              `provided`);
      this.modelURL = modelURL;
      this.metadataURL = metadataURL;
    }

    this.parameters = {
      sampleRateHz: this.SAMPLE_RATE_HZ,
      fftSize: this.FFT_SIZE
    };
  }

  /**
   * Start streaming recognition.
   *
   * To stop the recognition, use `stopListening()`.
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
  async listen(
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
    let probabilityThreshold =
        config.probabilityThreshold == null ? 0 : config.probabilityThreshold;
    if (config.includeEmbedding) {
      // Override probability threshold to 0 if includeEmbedding is true.
      probabilityThreshold = 0;
    }
    tf.util.assert(
        probabilityThreshold >= 0 && probabilityThreshold <= 1,
        `Invalid probabilityThreshold value: ${probabilityThreshold}`);
    let invokeCallbackOnNoiseAndUnknown =
        config.invokeCallbackOnNoiseAndUnknown == null ?
        false :
        config.invokeCallbackOnNoiseAndUnknown;
    if (config.includeEmbedding) {
      // Override invokeCallbackOnNoiseAndUnknown threshold to true if
      // includeEmbedding is true.
      invokeCallbackOnNoiseAndUnknown = true;
    }

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

    const spectrogramCallback: SpectrogramCallback = async (x: tf.Tensor) => {
      const normalizedX = normalize(x);
      let y: tf.Tensor;
      let embedding: tf.Tensor;
      if (config.includeEmbedding) {
        await this.ensureModelWithEmbeddingOutputCreated();
        [y, embedding] =
            this.modelWithEmbeddingOutput.predict(normalizedX) as tf.Tensor[];
      } else {
        y = this.model.predict(normalizedX) as tf.Tensor;
      }

      const scores = await y.data() as Float32Array;
      const maxIndexTensor = y.argMax(-1);
      const maxIndex = (await maxIndexTensor.data())[0];
      const maxScore = Math.max(...scores);
      tf.dispose([y, maxIndexTensor, normalizedX]);

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
          callback({scores, spectrogram, embedding});
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
      numFramesPerSpectrogram: this.nonBatchInputShape[0],
      columnTruncateLength: this.nonBatchInputShape[1],
      suppressionTimeMillis,
      spectrogramCallback,
      overlapFactor
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

    const model = await tf.loadModel(this.modelURL);
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
        this.parameters.fftSize / this.parameters.sampleRateHz * 1e3;
    const numFrames = model.inputs[0].shape[1];
    this.parameters.spectrogramDurationMillis = numFrames * frameDurationMillis;
  }

  /**
   * Construct a two-output model that includes the following outputs:
   *
   * 1. The same softmax probability output as the original model's output
   * 2. The embedding, i.e., activation from the second-last dense layer of
   *    the original model.
   */
  protected async ensureModelWithEmbeddingOutputCreated() {
    if (this.modelWithEmbeddingOutput != null) {
      return;
    }
    await this.ensureModelLoaded();

    // Find the second last dense layer of the original model.
    let secondLastDenseLayer: tf.layers.Layer;
    for (let i = this.model.layers.length - 2; i >= 0; --i) {
      if (this.model.layers[i].getClassName() === 'Dense') {
        secondLastDenseLayer = this.model.layers[i];
        break;
      }
    }
    if (secondLastDenseLayer == null) {
      throw new Error(
          'Failed to find second last dense layer in the original model.');
    }
    this.modelWithEmbeddingOutput = tf.model({
      inputs: this.model.inputs,
      outputs: [
        this.model.outputs[0], secondLastDenseLayer.output as tf.SymbolicTensor
      ]
    });
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
    const metadataJSON = await loadMetadataJson(this.metadataURL);
    this.words = metadataJSON.words;
  }

  /**
   * Stop streaming recognition.
   *
   * @throws Error if there is not ongoing streaming recognition.
   */
  async stopListening(): Promise<void> {
    if (!streaming) {
      throw new Error('Cannot stop streaming when streaming is not ongoing.');
    }
    await this.audioDataExtractor.stop();
    streaming = false;
  }

  /**
   * Check if streaming recognition is ongoing.
   */
  isListening(): boolean {
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
          'ensureModelLoaded(), recognize(), or listen().');
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
   * @param config Optional configuration object.
   * @returns Result of the recognition, with the following field:
   *   scores:
   *   - A `Float32Array` if there is only one input exapmle.
   *   - An `Array` of `Float32Array`, if there are multiple input examples.
   */
  async recognize(input?: tf.Tensor|Float32Array, config?: RecognizeConfig):
      Promise<SpeechCommandRecognizerResult> {
    if (config == null) {
      config = {};
    }

    await this.ensureModelLoaded();

    if (input == null) {
      // If `input` is not provided, draw audio data from WebAudio and us it
      // for recognition.
      const spectrogramData = await this.recognizeOnline();
      input = spectrogramData.data;
    }

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

    const output: SpeechCommandRecognizerResult = {scores: null};
    if (config.includeEmbedding) {
      // Optional inclusion of embedding (internal activation).
      await this.ensureModelWithEmbeddingOutputCreated();
      const outAndEmbedding =
          this.modelWithEmbeddingOutput.predict(inputTensor) as tf.Tensor[];
      outTensor = outAndEmbedding[0];
      output.embedding = outAndEmbedding[1];
    } else {
      outTensor = this.model.predict(inputTensor) as tf.Tensor;
    }

    if (numExamples === 1) {
      output.scores = await outTensor.data() as Float32Array;
    } else {
      const unstacked = tf.unstack(outTensor) as tf.Tensor[];
      const scorePromises = unstacked.map(item => item.data());
      output.scores = await Promise.all(scorePromises) as Float32Array[];
      tf.dispose(unstacked);
    }

    if (config.includeSpectrogram) {
      output.spectrogram = {
        data: (input instanceof tf.Tensor ? await input.data() : input) as
            Float32Array,
        frameSize: this.nonBatchInputShape[1],
      };
    }

    return output;
  }

  private async recognizeOnline(): Promise<SpectrogramData> {
    return new Promise<SpectrogramData>((resolve, reject) => {
      const spectrogramCallback: SpectrogramCallback = async (x: tf.Tensor) => {
        const normalizedX = normalize(x);
        await this.audioDataExtractor.stop();
        resolve({
          data: await normalizedX.data() as Float32Array,
          frameSize: this.nonBatchInputShape[1],
        });
        normalizedX.dispose();
        return false;
      };
      this.audioDataExtractor = new BrowserFftFeatureExtractor({
        sampleRateHz: this.parameters.sampleRateHz,
        numFramesPerSpectrogram: this.nonBatchInputShape[0],
        columnTruncateLength: this.nonBatchInputShape[1],
        suppressionTimeMillis: 0,
        spectrogramCallback,
        overlapFactor: 0
      });
      this.audioDataExtractor.start();
    });
  }

  createTransfer(name: string): TransferSpeechCommandRecognizer {
    if (this.model == null) {
      throw new Error(
          'Model has not been loaded yet. Load model by calling ' +
          'ensureModelLoaded(), recognizer(), or listen().');
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

  protected freezeModel(): void {
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
  private dataset: Dataset;
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
    this.words = null;
    this.dataset = new Dataset();
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
    return new Promise<SpectrogramData>(resolve => {
      const spectrogramCallback: SpectrogramCallback = async (x: tf.Tensor) => {
        const normalizedX = normalize(x);
        this.dataset.addExample({
          label: word,
          spectrogram: {
            data: await normalizedX.data() as Float32Array,
            frameSize: this.nonBatchInputShape[1],
          }
        });
        normalizedX.dispose();
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
        numFramesPerSpectrogram: this.nonBatchInputShape[0],
        columnTruncateLength: this.nonBatchInputShape[1],
        suppressionTimeMillis: 0,
        spectrogramCallback,
        overlapFactor: 0
      });
      this.audioDataExtractor.start();
    });
  }

  /**
   * Clear all transfer learning examples collected so far.
   */
  clearExamples(): void {
    tf.util.assert(
        this.words != null && this.words.length > 0 && !this.dataset.empty(),
        `No transfer learning examples exist for model name ${this.name}`);
    this.dataset.clear();
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
    if (this.dataset.empty()) {
      throw new Error(
          `No examples have been collected for transfer-learning model ` +
          `named '${this.name}' yet.`);
    }
    return this.dataset.getExampleCounts();
  }

  /**
   * Get examples currently held by the transfer-learning recognizer.
   *
   * @param label Label requested.
   * @returns An array of `Example`s, along with their UIDs.
   */
  getExamples(label: string): Array<{uid: string, example: Example}> {
    return this.dataset.getExamples(label);
  }

  /**
   * Load an array of serialized examples.
   *
   * @param serialized The examples in their serialized format.
   * @param clearExisting Whether to clear the existing examples while
   *   performing the loading (default: false).
   */
  loadExamples(serialized: ArrayBuffer, clearExisting = false): void {
    const incomingDataset = new Dataset(serialized);
    if (clearExisting) {
      this.clearExamples();
    }

    const incomingVocab = incomingDataset.getVocabulary();
    for (const label of incomingVocab) {
      const examples = incomingDataset.getExamples(label);
      for (const example of examples) {
        this.dataset.addExample(example.example);
      }
    }

    this.collateTransferWords();
  }

  /** Serialize the existing examples. */
  serializeExamples(): ArrayBuffer {
    return this.dataset.serialize();
  }

  /**
   * Collect the vocabulary of this transfer-learned recognizer.
   *
   * The words are put in an alphabetically sorted order.
   */
  private collateTransferWords() {
    this.words = this.dataset.getVocabulary();
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
    const out = this.dataset.getSpectrogramsAsTensors();
    return {xs: out.xs, ys: out.ys as tf.Tensor};
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
  async train(config?: TransferLearnConfig):
      Promise<tf.History|[tf.History, tf.History]> {
    tf.util.assert(
        this.words != null && this.words.length > 0,
        `Cannot train transfer-learning model '${this.name}' because no ` +
            `transfer learning example has been collected.`);
    tf.util.assert(
        this.words.length > 1,
        `Cannot train transfer-learning model '${this.name}' because only ` +
            `1 word label ('${JSON.stringify(this.words)}') ` +
            `has been collected for transfer learning. Requires at least 2.`);
    if (config.fineTuningEpochs != null) {
      tf.util.assert(
          config.fineTuningEpochs >= 0 &&
              Number.isInteger(config.fineTuningEpochs),
          `If specified, fineTuningEpochs must be a non-negative integer, ` +
              `but received ${config.fineTuningEpochs}`);
    }

    if (config == null) {
      config = {};
    }

    if (this.model == null) {
      this.createTransferModelFromBaseModel();
    }

    // This layer needs to be frozen for the initial phase of the
    // transfer learning. During subsequent fine-tuning (if any), it will
    // be unfrozen.
    this.secondLastBaseDenseLayer.trainable = false;

    // Compile model for training.
    this.model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: config.optimizer || 'sgd',
      metrics: ['acc']
    });

    // Prepare the data.
    const {xs, ys} = this.collectTransferDataAsTensors();

    let trainXs: tf.Tensor;
    let trainYs: tf.Tensor;
    let valData: [tf.Tensor, tf.Tensor];
    try {
      if (config.validationSplit != null) {
        const splits = balancedTrainValSplit(xs, ys, config.validationSplit);
        trainXs = splits.trainXs;
        trainYs = splits.trainYs;
        valData = [splits.valXs, splits.valYs];
      } else {
        trainXs = xs;
        trainYs = ys;
      }

      const history = await this.model.fit(trainXs, trainYs, {
        epochs: config.epochs == null ? 20 : config.epochs,
        validationData: valData,
        batchSize: config.batchSize,
        callbacks: config.callback == null ? null : [config.callback]
      });

      if (config.fineTuningEpochs != null && config.fineTuningEpochs > 0) {
        // Fine tuning: unfreeze the second-last dense layer of the base model.
        const originalTrainableValue = this.secondLastBaseDenseLayer.trainable;
        this.secondLastBaseDenseLayer.trainable = true;

        // Recompile model after unfreezing layer.
        const fineTuningOptimizer: string|tf.Optimizer =
            config.fineTuningOptimizer == null ? 'sgd' :
                                                 config.fineTuningOptimizer;
        this.model.compile({
          loss: 'categoricalCrossentropy',
          optimizer: fineTuningOptimizer,
          metrics: ['acc']
        });

        const fineTuningHistory = await this.model.fit(trainXs, trainYs, {
          epochs: config.fineTuningEpochs,
          validationData: valData,
          batchSize: config.batchSize,
          callbacks: config.fineTuningCallback == null ?
              null :
              [config.fineTuningCallback]
        });

        // Set the trainable attribute of the fine-tuning layer to its
        // previous value.
        this.secondLastBaseDenseLayer.trainable = originalTrainableValue;
        return [history, fineTuningHistory];
      } else {
        return history;
      }
    } finally {
      tf.dispose([xs, ys, trainXs, trainYs, valData]);
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
    this.secondLastBaseDenseLayer = layers[layerIndex];
    const truncatedBaseOutput =
        this.secondLastBaseDenseLayer.output as tf.SymbolicTensor;

    this.transferHead = tf.sequential();
    this.transferHead.add(tf.layers.dense({
      units: this.words.length,
      activation: 'softmax',
      inputShape: truncatedBaseOutput.shape.slice(1)
    }));
    const transferOutput =
        this.transferHead.apply(truncatedBaseOutput) as tf.SymbolicTensor;
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
