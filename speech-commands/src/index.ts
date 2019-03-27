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

import {BrowserFftSpeechCommandRecognizer} from './browser_fft_recognizer';
import {concatenateFloat32Arrays} from './generic_utils';
import {FFT_TYPE, SpeechCommandRecognizer} from './types';

/**
 * Create an instance of speech-command recognizer.
 *
 * @param fftType Type of FFT. The currently availble option(s):
 *   - BROWSER_FFT: Obtains audio spectrograms using browser's native Fourier
 *     transform.
 * @param vocabulary The vocabulary of the model to load. Possible options:
 *   - '18w' (default): The 18-word vocaulbary, consisting of:
 *     'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
 *     'eight', 'nine', 'up', 'down', 'left', 'right', 'go', 'stop',
 *     'yes', and 'no', in addition to '_background_noise_' and '_unknown_'.
 *   - 'directional4w': The four directional words: 'up', 'down', 'left', and
 *     'right', in addition to '_background_noise_' and '_unknown_'.
 *   Choosing a smaller vocabulary leads to better accuracy on the words of
 *   interest and a slightly smaller model size.
 * @param customModelURL A custom model URL pointing to a model.json file.
 *   Supported schemes: http://, https://, and node.js-only: file://.
 *   Mutually exclusive with `vocabulary`. If provided, `customMetadatURL`
 *   most also be provided.
 * @param customMetadataURL A custom metadata URL pointing to a metadata.json
 *   file. Must be provided together with `customModelURL`.
 * @returns An instance of SpeechCommandRecognizer.
 * @throws Error on invalid value of `fftType`.
 */
export function create(
    fftType: FFT_TYPE, vocabulary?: string, customModelURL?: string,
    customMetadataURL?: string): SpeechCommandRecognizer {
  tf.util.assert(
      customModelURL == null && customMetadataURL == null ||
          customModelURL != null && customMetadataURL != null,
      () => `customModelURL and customMetadataURL must be both provided or ` +
          `both not provided.`);
  if (customModelURL != null) {
    tf.util.assert(
        vocabulary == null,
        () => `vocabulary name must be null or undefined when modelURL ` +
            `is provided.`);
  }

  if (fftType === 'BROWSER_FFT') {
    return new BrowserFftSpeechCommandRecognizer(
        vocabulary, customModelURL, customMetadataURL);
  } else if (fftType === 'SOFT_FFT') {
    throw new Error(
        'SOFT_FFT SpeechCommandRecognizer has not been implemented yet.');
  } else {
    throw new Error(`Invalid fftType: '${fftType}'`);
  }
}

const utils = {concatenateFloat32Arrays};

export {BACKGROUND_NOISE_TAG, Dataset, GetDataConfig as GetSpectrogramsAsTensorsConfig, getMaxIntensityFrameIndex, spectrogram2IntensityCurve, SpectrogramAndTargetsTfDataset} from './dataset';
export {Example, FFT_TYPE, RecognizerParams, SpectrogramData, SpeechCommandRecognizer, SpeechCommandRecognizerMetadata, SpeechCommandRecognizerResult, StreamingRecognitionConfig, TransferLearnConfig, TransferSpeechCommandRecognizer} from './types';
export {deleteSavedTransferModel, listSavedTransferModels, UNKNOWN_TAG} from './browser_fft_recognizer';
export {utils};
export {version} from './version';
