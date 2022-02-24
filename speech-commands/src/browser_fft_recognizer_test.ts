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

import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as tfl from '@tensorflow/tfjs-layers';
import {writeFileSync} from 'fs';
import {join} from 'path';
import * as rimraf from 'rimraf';
import * as tempfile from 'tempfile';

import {BrowserFftSpeechCommandRecognizer, deleteSavedTransferModel, getMajorAndMinorVersion, listSavedTransferModels, localStorageWrapper, SAVED_MODEL_METADATA_KEY} from './browser_fft_recognizer';
import * as BrowserFftUtils from './browser_fft_utils';
import {FakeAudioContext, FakeAudioMediaStream} from './browser_test_utils';
import {arrayBuffer2SerializedExamples, BACKGROUND_NOISE_TAG} from './dataset';
import {create} from './index';
import {SpeechCommandRecognizerResult} from './types';
import {version} from './version';

describe('getMajorAndMinorVersion', () => {
  it('Correct results', () => {
    expect(getMajorAndMinorVersion('0.1.3')).toEqual('0.1');
    expect(getMajorAndMinorVersion('1.0.9')).toEqual('1.0');
    expect(getMajorAndMinorVersion('2.0.0rc0')).toEqual('2.0');
    expect(getMajorAndMinorVersion('2.0.9999999')).toEqual('2.0');
    expect(getMajorAndMinorVersion('3.0')).toEqual('3.0');
  });
});

describeWithFlags('Browser FFT recognizer', NODE_ENVS, () => {
  const fakeWords: string[] = [
    '_background_noise_', 'down', 'eight', 'five', 'four', 'go', 'left', 'nine',
    'one', 'right', 'seven', 'six', 'stop', 'three', 'two', 'up', 'zero'
  ];
  const fakeWordsNoiseAndUnknownOnly: string[] =
      ['_background_noise_', '_unknown_'];

  const fakeNumFrames = 42;
  const fakeColumnTruncateLength = 232;

  let secondLastBaseDenseLayer: tfl.layers.Layer;
  let tfLoadModelSpy: jasmine.Spy;

  function setUpFakes(model?: tfl.Sequential, backgroundAndNoiseOnly = false) {
    const words =
        backgroundAndNoiseOnly ? fakeWordsNoiseAndUnknownOnly : fakeWords;
    const numWords = words.length;
    tfLoadModelSpy =
        spyOn(tfl, 'loadLayersModel').and.callFake((url: string) => {
          if (model == null) {
            model = tfl.sequential();
            model.add(tfl.layers.flatten(
                {inputShape: [fakeNumFrames, fakeColumnTruncateLength, 1]}));
            secondLastBaseDenseLayer = tfl.layers.dense({
              units: 4,
              activation: 'relu',
              kernelInitializer: 'leCunNormal'
            });
            model.add(secondLastBaseDenseLayer);
            model.add(tfl.layers.dense({
              units: numWords,
              useBias: false,
              kernelInitializer: 'leCunNormal',
              activation: 'softmax'
            }));
          }
          return model;
        });
    spyOn(BrowserFftUtils, 'loadMetadataJson')
        .and.callFake(async (url: string) => {
          return {words};
        });

    spyOn(BrowserFftUtils, 'getAudioContextConstructor')
        .and.callFake(() => FakeAudioContext.createInstance);
    spyOn(BrowserFftUtils, 'getAudioMediaStream')
        .and.callFake(() => new FakeAudioMediaStream());
  }

  it('Constructor works', () => {
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    expect(recognizer.isListening()).toEqual(false);
    expect(recognizer.params().sampleRateHz).toEqual(44100);
    expect(recognizer.params().fftSize).toEqual(1024);
  });

  it('ensureModelLoaded succeeds', async () => {
    setUpFakes();

    const recognizer = new BrowserFftSpeechCommandRecognizer();
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual(fakeWords);
    expect(recognizer.model instanceof tfl.LayersModel).toEqual(true);
    expect(recognizer.modelInputShape()).toEqual([
      null, fakeNumFrames, fakeColumnTruncateLength, 1
    ]);
  });

  it('ensureModelLoaded fails: words - model output mismatch', async () => {
    const fakeModel = tfl.sequential();
    fakeModel.add(tfl.layers.flatten(
        {inputShape: [fakeNumFrames, fakeColumnTruncateLength, 1]}));
    fakeModel.add(tfl.layers.dense({units: 12, activation: 'softmax'}));
    setUpFakes(fakeModel);

    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;
    try {
      await recognizer.ensureModelLoaded();
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Mismatch between .* dimension.*12.*17/);
  });

  async function createFakeModelArtifact(tmpDir: string) {
    const model = tfl.sequential();
    model.add(tfl.layers.reshape(
        {targetShape: [43 * 232], inputShape: [43, 232, 1]}));
    model.add(tfl.layers.dense({units: 4, activation: 'softmax'}));
    await model.save(`file://${tmpDir}`);
  }

  function createFakeMetadataFile(tmpDir: string) {
    // Construct the metadata.json for the fake model.
    const metadata: {} = {
      wordLabels: ['_background_noise_', '_unknown_', 'foo', 'bar'],
      frameSize: 232
    };
    const metadataPath = join(tmpDir, 'metadata.json');
    writeFileSync(metadataPath, JSON.stringify(metadata));
  }

  function createFakeMetadataFileWithLegacyWordsField(tmpDir: string) {
    // Construct the metadata.json for the fake model.
    const metadata: {} = {
      words: ['_background_noise_', '_unknown_', 'foo', 'bar'],
      frameSize: 232
    };
    const metadataPath = join(tmpDir, 'metadata.json');
    writeFileSync(metadataPath, JSON.stringify(metadata));
  }

  it('Constructing recognizer: custom URLs', async () => {
    // Construct a fake model
    const tmpDir = tempfile();
    await createFakeModelArtifact(tmpDir);
    createFakeMetadataFile(tmpDir);

    const modelPath = join(tmpDir, 'model.json');
    const metadataPath = join(tmpDir, 'metadata.json');
    const modelURL = `file://${modelPath}`;
    const metadataURL = `file://${metadataPath}`;

    const recognizer =
        new BrowserFftSpeechCommandRecognizer(null, modelURL, metadataURL);
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual([
      '_background_noise_', '_unknown_', 'foo', 'bar'
    ]);

    const recogResult = await recognizer.recognize(tf.zeros([2, 43, 232, 1]));
    expect(recogResult.scores.length).toEqual(2);
    expect((recogResult.scores[0] as Float32Array).length).toEqual(4);
    expect((recogResult.scores[1] as Float32Array).length).toEqual(4);

    rimraf(tmpDir, () => {});
  });

  it('Constructing recognizer: custom URLs, legacy words format', async () => {
    // Construct a fake model
    const tmpDir = tempfile();
    await createFakeModelArtifact(tmpDir);
    createFakeMetadataFileWithLegacyWordsField(tmpDir);

    const modelPath = join(tmpDir, 'model.json');
    const metadataPath = join(tmpDir, 'metadata.json');
    const modelURL = `file://${modelPath}`;
    const metadataURL = `file://${metadataPath}`;

    const recognizer =
        new BrowserFftSpeechCommandRecognizer(null, modelURL, metadataURL);
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual([
      '_background_noise_', '_unknown_', 'foo', 'bar'
    ]);

    const recogResult = await recognizer.recognize(tf.zeros([2, 43, 232, 1]));
    expect(recogResult.scores.length).toEqual(2);
    expect((recogResult.scores[0] as Float32Array).length).toEqual(4);
    expect((recogResult.scores[1] as Float32Array).length).toEqual(4);

    rimraf(tmpDir, () => {});
  });

  it('Creating recognizer using custom URLs', async () => {
    // Construct a fake model
    const tmpDir = tempfile();
    await createFakeModelArtifact(tmpDir);
    createFakeMetadataFile(tmpDir);

    const modelPath = join(tmpDir, 'model.json');
    const metadataPath = join(tmpDir, 'metadata.json');
    const modelURL = `file://${modelPath}`;
    const metadataURL = `file://${metadataPath}`;

    const recognizer = create('BROWSER_FFT', null, modelURL, metadataURL);
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual([
      '_background_noise_', '_unknown_', 'foo', 'bar'
    ]);

    const recogResult = await recognizer.recognize(tf.zeros([2, 43, 232, 1]));
    expect(recogResult.scores.length).toEqual(2);
    expect((recogResult.scores[0] as Float32Array).length).toEqual(4);
    expect((recogResult.scores[1] as Float32Array).length).toEqual(4);

    rimraf(tmpDir, () => {});
  });

  it('Providing both vocabulary and modelURL leads to Error', () => {
    expect(
        () => new BrowserFftSpeechCommandRecognizer(
            'vocab_1', 'http://localhost/model.json',
            'http://localhost/metadata.json'))
        .toThrowError(/vocabulary name must be null or undefined .* modelURL/);
  });

  it('Providing modelURL without metadataURL leads to Error', () => {
    expect(
        () => new BrowserFftSpeechCommandRecognizer(
            null, 'http://localhost/model.json'))
        .toThrowError(/modelURL and metadataURL must be both provided/);
  });

  it('Offline recognize succeeds with single tf.Tensor', async () => {
    setUpFakes();

    const spectrogram =
        tf.zeros([1, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(output.scores instanceof Float32Array).toEqual(true);
    expect(output.scores.length).toEqual(17);
  });

  it('Offline recognize succeeds with batched tf.Tensor', async () => {
    setUpFakes();

    const spectrogram =
        tf.zeros([3, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(Array.isArray(output.scores)).toEqual(true);
    expect(output.scores.length).toEqual(3);
    for (let i = 0; i < 3; ++i) {
      expect((output.scores[i] as Float32Array).length).toEqual(17);
    }
  });

  it('Offline recognize call: includeEmbedding', async () => {
    setUpFakes();

    // A batch of examples.
    const numExamples = 3;
    const spectrogram =
        tf.zeros([numExamples, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    // Warm-up recognize call, for subsequent memory-leak check.
    await recognizer.recognize(spectrogram, {includeEmbedding: true});
    const numTensors0 = tf.memory().numTensors;  // For memory-leak check.
    const output =
        await recognizer.recognize(spectrogram, {includeEmbedding: true});
    expect(Array.isArray(output.scores)).toEqual(true);
    expect(output.scores.length).toEqual(3);
    for (let i = 0; i < 3; ++i) {
      expect((output.scores[i] as Float32Array).length).toEqual(17);
    }
    expect(output.embedding.rank).toEqual(2);
    expect(output.embedding.shape[0]).toEqual(numExamples);
    tf.dispose(output.embedding);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
  });

  it('Offline recognize fails due to incorrect shape', async () => {
    setUpFakes();

    const spectrogram =
        tf.zeros([1, fakeNumFrames, fakeColumnTruncateLength, 2]);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;
    try {
      await recognizer.recognize(spectrogram);
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected .* shape .*, but got shape/);
  });

  it('Offline recognize succeeds with single Float32Array', async () => {
    setUpFakes();

    const spectrogram =
        new Float32Array(fakeNumFrames * fakeColumnTruncateLength * 1);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(output.scores instanceof Float32Array).toEqual(true);
    expect(output.scores.length).toEqual(17);
  });

  it('Offline recognize succeeds with batched Float32Array', async () => {
    setUpFakes();

    const spectrogram =
        new Float32Array(2 * fakeNumFrames * fakeColumnTruncateLength * 1);
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(Array.isArray(output.scores)).toEqual(true);
    expect(output.scores.length).toEqual(2);
    for (let i = 0; i < 2; ++i) {
      expect((output.scores[i] as Float32Array).length).toEqual(17);
    }
  });

  it('listen() call with invalid overlapFactor', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;

    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: -1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);

    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: 1});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);

    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: 1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);
  });

  it('listen() call with invalid probabilityThreshold', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;
    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {},
          {probabilityThreshold: 1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Invalid probabilityThreshold value: 1\.2/);

    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {},
          {probabilityThreshold: -0.1});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Invalid probabilityThreshold value: -0\.1/);
  });

  it('streaming: overlapFactor = 0', done => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    const numCallbacksToComplete = 2;
    let numCallbacksCompleted = 0;
    const spectroDurationMillis = 900;
    const tensorCounts: number[] = [];
    const callbackTimestamps: number[] = [];
    recognizer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(fakeWords.length);

      callbackTimestamps.push(tf.util.now());
      if (callbackTimestamps.length > 1) {
        const timeBetweenCallbacks =
            callbackTimestamps[callbackTimestamps.length - 1] -
            callbackTimestamps[callbackTimestamps.length - 2];
        expect(
            timeBetweenCallbacks > spectroDurationMillis &&
            timeBetweenCallbacks < 1.3 * spectroDurationMillis)
            .toBe(true);
      }

      tensorCounts.push(tf.memory().numTensors);
      if (tensorCounts.length > 1) {
        // Assert no memory leak.
        expect(tensorCounts[tensorCounts.length - 1])
            .toEqual(tensorCounts[tensorCounts.length - 2]);
      }

      // spectrogram is not provided by default.
      expect(result.spectrogram).toBeUndefined();

      // Embedding should not be included by default.
      expect(result.embedding).toBeUndefined();

      if (++numCallbacksCompleted >= numCallbacksToComplete) {
        await recognizer.stopListening();
        done();
      }
    }, {overlapFactor: 0, invokeCallbackOnNoiseAndUnknown: true});
  });

  it('streaming: overlapFactor = 0, includeEmbedding', done => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    const numCallbacksToComplete = 2;
    let numCallbacksCompleted = 0;
    const tensorCounts: number[] = [];
    const callbackTimestamps: number[] = [];
    recognizer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(fakeWords.length);

      callbackTimestamps.push(tf.util.now());
      const timeDelta = 50;
      if (callbackTimestamps.length > 1) {
        expect(
            callbackTimestamps[callbackTimestamps.length - 1] -
            callbackTimestamps[callbackTimestamps.length - 2])
            .toBeGreaterThanOrEqual(
                recognizer.params().spectrogramDurationMillis - timeDelta);
      }

      tensorCounts.push(tf.memory().numTensors);

      // spectrogram is not provided by default.
      expect(result.spectrogram).toBeUndefined();

      // Embedding should not be included by default.
      expect(result.embedding.rank).toEqual(2);
      expect(result.embedding.shape[0]).toEqual(1);
      // The number of units of the hidden dense layer.
      expect(result.embedding.shape[1]).toEqual(4);

      if (++numCallbacksCompleted >= numCallbacksToComplete) {
        await recognizer.stopListening();
        done();
      }
    }, {
      overlapFactor: 0,
      invokeCallbackOnNoiseAndUnknown: true,
      includeEmbedding: true
    });
  });

  it('streaming: overlapFactor = 0.5, includeSpectrogram', done => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    const numCallbacksToComplete = 2;
    let numCallbacksCompleted = 0;
    const spectroDurationMillis = 900;
    const tensorCounts: number[] = [];
    const callbackTimestamps: number[] = [];
    recognizer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(fakeWords.length);

      callbackTimestamps.push(tf.util.now());
      if (callbackTimestamps.length > 1) {
        const timeBetweenCallbacks =
            callbackTimestamps[callbackTimestamps.length - 1] -
            callbackTimestamps[callbackTimestamps.length - 2];
        expect(
            timeBetweenCallbacks > 0.5 * spectroDurationMillis &&
            timeBetweenCallbacks < 0.8 * spectroDurationMillis)
            .toBe(true);
      }

      tensorCounts.push(tf.memory().numTensors);
      if (tensorCounts.length > 1) {
        // Assert no memory leak.
        expect(tensorCounts[tensorCounts.length - 1])
            .toEqual(tensorCounts[tensorCounts.length - 2]);
      }

      // spectrogram is not provided by default.
      expect(result.spectrogram.data.length)
          .toBe(fakeNumFrames * fakeColumnTruncateLength);
      expect(result.spectrogram.frameSize).toBe(fakeColumnTruncateLength);

      if (++numCallbacksCompleted >= numCallbacksToComplete) {
        await recognizer.stopListening();
        done();
      }
    }, {
      overlapFactor: 0.5,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
  });

  it('streaming: invokeCallbackOnNoiseAndUnknown = false', done => {
    setUpFakes(null, true);
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    let callbackInvokeCount = 0;
    recognizer.listen(async (result: SpeechCommandRecognizerResult) => {
      callbackInvokeCount++;
    }, {overlapFactor: 0.5, invokeCallbackOnNoiseAndUnknown: false});

    setTimeout(() => {
      recognizer.stopListening();
      // Due to `invokeCallbackOnNoiseAndUnknown: false` and the fact that the
      // vocabulary contains only _background_noise_ and _unknown_, the callback
      // should have never been called.
      expect(callbackInvokeCount).toEqual(0);
      done();
    }, 1000);
  });

  it('streaming: invokeCallbackOnNoiseAndUnknown = true', done => {
    setUpFakes(null, true);
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    let callbackInvokeCount = 0;
    recognizer.listen(async (result: SpeechCommandRecognizerResult) => {
      callbackInvokeCount++;
    }, {overlapFactor: 0.5, invokeCallbackOnNoiseAndUnknown: true});

    setTimeout(() => {
      recognizer.stopListening();
      // Even though the model predicts only _background_noise_ and _unknown_,
      // the callback should have been invoked because of
      // `invokeCallbackOnNoiseAndUnknown: true`.
      expect(callbackInvokeCount).toBeGreaterThan(0);
      done();
    }, 1000);
  });

  it('Attempt to start streaming twice leads to Error', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    await recognizer.listen(
        async (result: SpeechCommandRecognizerResult) => {});
    expect(recognizer.isListening()).toEqual(true);

    let caughtError: Error;
    try {
      await recognizer.listen(
          async (result: SpeechCommandRecognizerResult) => {});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toEqual('Cannot start streaming again when streaming is ongoing.');
    expect(recognizer.isListening()).toEqual(true);

    await recognizer.stopListening();
    expect(recognizer.isListening()).toEqual(false);
  });

  it('Attempt to stop streaming twice leads to Error', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    await recognizer.listen(
        async (result: SpeechCommandRecognizerResult) => {});
    expect(recognizer.isListening()).toEqual(true);

    await recognizer.stopListening();
    expect(recognizer.isListening()).toEqual(false);

    let caughtError: Error;
    try {
      await recognizer.stopListening();
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toEqual('Cannot stop streaming when streaming is not ongoing.');
    expect(recognizer.isListening()).toEqual(false);
  });

  it('Online recognize() call succeeds', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    for (let i = 0; i < 2; ++i) {
      // No-arg call: online recognition.
      const output = await recognizer.recognize();
      expect(output.scores.length).toEqual(fakeWords.length);
      expect(output.embedding).toBeUndefined();
    }
  });

  it('Online recognize() call with includeEmbedding succeeds', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    for (let i = 0; i < 2; ++i) {
      // No-arg call: online recognition.
      const output = await recognizer.recognize(null, {includeEmbedding: true});
      expect(output.scores.length).toEqual(fakeWords.length);
      expect(output.embedding.rank).toEqual(2);
      expect(output.embedding.shape[0]).toEqual(1);
      expect(output.spectrogram).toBeUndefined();
    }
  });

  it('Online recognize() call with includeSpectrogram succeeds', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    for (let i = 0; i < 2; ++i) {
      // No-arg call: online recognition.
      const output =
          await recognizer.recognize(null, {includeSpectrogram: true});
      expect(output.scores.length).toEqual(fakeWords.length);
      expect(output.embedding).toBeUndefined();
      expect(output.spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
      expect(output.spectrogram.data.length)
          .toEqual(fakeColumnTruncateLength * fakeNumFrames);
    }
  });

  it('collectExample with durationSec', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const params = transfer.params();
    // Double the length of the spectrogram.
    const durationSec = params.spectrogramDurationMillis * 2 / 1e3;
    const spectrogram = await transfer.collectExample('foo', {durationSec});
    expect(spectrogram.data.length / fakeColumnTruncateLength / fakeNumFrames)
        .toEqual(2);
    const example = transfer.getExamples('foo')[0];
    expect(example.example.rawAudio).toBeUndefined();
  });

  it('collectExample with 0 durationSec errors', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 0;
    try {
      await transfer.collectExample('foo', {durationSec});
      done.fail('Failed to catch expected error');
    } catch (err) {
      expect(err.message).toMatch(/Expected durationSec to be > 0/);
      done();
    }
  });

  it('collectExample: durationMultiplier&durationSec errors', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1;
    const durationMultiplier = 2;
    try {
      await transfer.collectExample('foo', {durationSec, durationMultiplier});
      done.fail('Failed to catch expected error');
    } catch (err) {
      expect(err.message).toMatch(/are mutually exclusive/);
      done();
    }
  });

  it('collectExample with onSnippet', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1;
    const snippetDurationSec = 0.1;
    const snippetLengths: number[] = [];
    const finalSpectrogram = await transfer.collectExample('foo', {
      durationSec,
      snippetDurationSec,
      onSnippet: async spectrogram => {
        snippetLengths.push(spectrogram.data.length);
      }
    });
    expect(snippetLengths.length).toEqual(11);
    expect(snippetLengths[0]).toEqual(927);
    // First audio sample is zero and should have been skipped.
    for (let i = 1; i < snippetLengths.length; ++i) {
      expect(snippetLengths[i]).toEqual(928);
    }
    expect(finalSpectrogram.data.length)
        .toEqual(snippetLengths.reduce((x, prev) => x + prev));
    expect(finalSpectrogram.data.length).toEqual(10208 - 1);
  });

  it('collectExample w/ invalid durationSec leads to error', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1;
    const snippetDurationSec = 0;
    try {
      await transfer.collectExample('foo', {durationSec, snippetDurationSec});
      done.fail();
    } catch (error) {
      expect(error.message).toMatch(/snippetDurationSec is expected to be > 0/);
      done();
    }
  });

  it('collectExample w/ onSnippet w/o snippetDurationSec error', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1;
    try {
      await transfer.collectExample(
          'foo', {durationSec, onSnippet: async spectrogram => {}});
      done.fail();
    } catch (error) {
      expect(error.message)
          .toMatch(/snippetDurationSec must be provided if onSnippet/);
      done();
    }
  });

  it('collectExample w/ snippetDurationSec w/o callback errors', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1;
    const snippetDurationSec = 0.1;
    try {
      await transfer.collectExample('foo', {durationSec, snippetDurationSec});
      done.fail();
    } catch (error) {
      expect(error.message)
          .toMatch(/onSnippet must be provided if snippetDurationSec/);
      done();
    }
  });

  it('collectExample: includeRawAudio, no snippets', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1.5;
    const includeRawAudio = true;
    await transfer.collectExample('foo', {durationSec, includeRawAudio});
    const examples = transfer.getExamples('foo');
    expect(examples.length).toEqual(1);
    expect(examples[0].example.rawAudio.sampleRateHz).toEqual(44100);
    expect(examples[0].example.rawAudio.data.length / (durationSec * 44100))
        .toBeCloseTo(1, 1e-3);
  });

  it('collectExample: includeRawAudio, with snippets', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const durationSec = 1.5;
    const snippetDurationSec = 0.1;
    const includeRawAudio = true;
    await transfer.collectExample('foo', {
      durationSec,
      includeRawAudio,
      snippetDurationSec,
      onSnippet: async spectrogram => {}
    });
    const examples = transfer.getExamples('foo');
    expect(examples.length).toEqual(1);
    expect(examples[0].example.rawAudio.sampleRateHz).toEqual(44100);
    expect(examples[0].example.rawAudio.data.length / (durationSec * 44100))
        .toBeCloseTo(1, 1e-3);
  });

  it('collectTransferLearningExample default transfer model', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    let spectrogram = await transfer.collectExample('foo');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer.wordLabels()).toEqual(['foo']);
    // Assert no cross-talk.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.countExamples()).toEqual({'foo': 1});

    spectrogram = await transfer.collectExample('foo');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer.wordLabels()).toEqual(['foo']);
    expect(transfer.countExamples()).toEqual({'foo': 2});

    spectrogram = await transfer.collectExample('bar');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);
    expect(transfer.countExamples()).toEqual({'bar': 1, 'foo': 2});
  });

  it('createTransfer with invalid name leads to Error', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    expect(() => base.createTransfer('')).toThrowError(/non-empty string/);
    expect(() => base.createTransfer(null)).toThrowError(/non-empty string/);
    expect(() => base.createTransfer(undefined))
        .toThrowError(/non-empty string/);
  });

  it('createTransfer with duplicate name leads to Error', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    base.createTransfer('xfer1');
    expect(() => base.createTransfer('xfer1'))
        .toThrowError(
            /There is already a transfer-learning model named \'xfer1\'/);
    base.createTransfer('xfer2');
  });

  it('createTransfer before model loading leads to Error', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    expect(() => base.createTransfer('xfer1'))
        .toThrowError(/Model has not been loaded yet/);
  });

  it('transfer recognizer has correct modelInputShape', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    expect(transfer.modelInputShape()).toEqual(base.modelInputShape());
  });

  it('transfer recognizer has correct params', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    expect(transfer.params()).toEqual(base.params());
  });

  it('clearTransferLearningExamples default transfer model', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    let spectrogram = await transfer.collectExample('foo');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer.wordLabels()).toEqual(['foo']);
    // Assert no cross-talk.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.countExamples()).toEqual({'foo': 1});

    transfer.clearExamples();
    expect(transfer.wordLabels()).toEqual(null);
    expect(() => transfer.countExamples()).toThrow();

    spectrogram = await transfer.collectExample('bar');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer.wordLabels()).toEqual(['bar']);
    expect(transfer.countExamples()).toEqual({'bar': 1});
  });

  it('Collect examples for 2 transfer models', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = base.createTransfer('xfer1');
    let spectrogram = await transfer1.collectExample('foo');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer1.wordLabels()).toEqual(['foo']);

    const transfer2 = await base.createTransfer('xfer2');
    spectrogram = await transfer2.collectExample('bar');
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    expect(spectrogram.data.length)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength);
    expect(transfer2.wordLabels()).toEqual(['bar']);
    expect(transfer1.wordLabels()).toEqual(['foo']);

    transfer1.clearExamples();
    expect(transfer2.wordLabels()).toEqual(['bar']);
    expect(transfer1.wordLabels()).toEqual(null);
    // Assert no cross-talk.
    expect(base.wordLabels()).toEqual(fakeWords);
  });

  it('clearExamples fails if called without examples', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    expect(() => transfer.clearExamples())
        .toThrowError(/No transfer learning examples .*xfer1/);
  });

  it('collectExample fails on undefined/null/empty word', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    let errorCaught: Error;
    try {
      await transfer.collectExample(undefined);
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message).toMatch(/non-empty string/);
    try {
      await transfer.collectExample(null);
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message).toMatch(/non-empty string/);
    try {
      await transfer.collectExample('');
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message).toMatch(/non-empty string/);
  });

  it('Concurrent collectTransferLearningExample call fails', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = await base.createTransfer('xfer1');
    transfer1.collectExample('foo').then(() => {
      done();
    });

    let caughtError: Error;
    try {
      await transfer1.collectExample('foo');
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Cannot start collection of transfer-learning example/);
  });

  it('Concurrent collectExample+listen fails', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    await base.listen(async (result: SpeechCommandRecognizerResult) => {});
    expect(base.isListening()).toEqual(true);

    const transfer = base.createTransfer('xfer1');
    // Concurrent with the ongoing listening (started by the listen() call
    // above).
    const example = await transfer.collectExample('foo');
    expect(example.frameSize).toEqual(232);

    await base.stopListening();
    expect(base.isListening()).toEqual(false);
  });

  it('trainTransferLearningModel default params', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    for (let i = 0; i < 2; ++i) {
      await transfer.collectExample('bar');
    }

    // Train transfer-learning model once to make sure model is created
    // first, so that we can check the change in the transfer-learning model's
    // weights after a new round of training.
    await transfer.train({epochs: 1, optimizer: tf.train.sgd(0)});

    const baseModel = base.model;
    // Assert that the base model has been frozen.
    for (const layer of baseModel.layers) {
      expect(layer.trainable).toEqual(false);
    }

    const baseModelOldWeightValues: Float32Array[] = [];
    baseModel.layers.forEach(layer => {
      layer.getWeights().forEach(w => {
        baseModelOldWeightValues.push(w.dataSync() as Float32Array);
      });
    });

    // tslint:disable-next-line:no-any
    const transferHead = (transfer as any).transferHead as tfl.Sequential;
    const numLayers = transferHead.layers.length;
    const oldTransferWeightValues = transferHead.getLayer(null, numLayers - 1)
                                        .getWeights()
                                        .map(weight => weight.dataSync());

    const history =
        await transfer.train({optimizer: tf.train.sgd(1)}) as tfl.History;
    expect(history.history.loss.length).toEqual(20);
    expect(history.history.acc.length).toEqual(20);

    const baseModelNewWeightValues: Float32Array[] = [];
    baseModel.layers.forEach(layer => {
      layer.getWeights().forEach(w => {
        baseModelNewWeightValues.push(w.dataSync() as Float32Array);
      });
    });

    // Verify that the weights of the dense layer in the base model doesn't
    // change, i.e., is frozen.
    const newTransferWeightValues = transferHead.getLayer(null, numLayers - 1)
                                        .getWeights()
                                        .map(weight => weight.dataSync());
    baseModelOldWeightValues.forEach((oldWeight, i) => {
      tf.test_util.expectArraysClose(baseModelNewWeightValues[i], oldWeight);
    });
    // Verify that the weight of the transfer-learning head model changes
    // after training.
    const maxWeightChanges = newTransferWeightValues.map(
        (newValues, i) => tf.max(tf.abs(tf.sub(
                                     tf.tensor1d(newValues),
                                     tf.tensor1d(oldTransferWeightValues[i]))))
                              .dataSync()[0]);
    expect(Math.max(...maxWeightChanges)).toBeGreaterThan(1e-3);

    // Test recognize() with the transfer recognizer.
    const spectrogram =
        tf.zeros([1, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const result = await transfer.recognize(spectrogram);
    expect(result.scores.length).toEqual(2);

    // After the transfer learning is complete, listen() with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);
    transfer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      await transfer.stopListening();
      done();
    });
  });

  it('trainTransferLearningModel custom params', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    for (let i = 0; i < 2; ++i) {
      await transfer.collectExample('bar');
    }
    const history =
        await transfer.train({epochs: 10, batchSize: 2}) as tfl.History;
    expect(history.history.loss.length).toEqual(10);
    expect(history.history.acc.length).toEqual(10);

    // After the transfer learning is complete, listen() with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);
    transfer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      await transfer.stopListening();
      done();
    });
  });

  it('trainTransferLearningModel w/ mixing-noise augmentation', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    for (let i = 0; i < 2; ++i) {
      await transfer.collectExample(BACKGROUND_NOISE_TAG);
    }
    const history =
        await transfer.train(
            {epochs: 10, batchSize: 2, augmentByMixingNoiseRatio: 0.5}) as
        tfl.History;
    expect(history.history.loss.length).toEqual(10);
    expect(history.history.acc.length).toEqual(10);

    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual([BACKGROUND_NOISE_TAG, 'foo']);
  });

  it('train and evaluate', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('_background_noise_');
    await transfer.collectExample('bar');
    await transfer.collectExample('bar');
    await transfer.train({epochs: 3, batchSize: 2, validationSplit: 0.5});

    const wordProbThresholds = [0, 0.25, 0.5, 0.75, 1];
    // Burn-in run for evaluate() memory tracking:
    await transfer.evaluate({windowHopRatio: 0.25, wordProbThresholds});
    const numTensors0 = tf.memory().numTensors;
    const {rocCurve, auc} =
        await transfer.evaluate({windowHopRatio: 0.25, wordProbThresholds});
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
    expect(rocCurve.length).toEqual(wordProbThresholds.length);
    for (let i = 0; i < rocCurve.length; ++i) {
      expect(rocCurve[i].probThreshold).toEqual(wordProbThresholds[i]);
      expect(rocCurve[i].fpr).toBeGreaterThanOrEqual(0);
      expect(rocCurve[i].fpr).toBeLessThanOrEqual(1);
      expect(rocCurve[i].tpr).toBeGreaterThanOrEqual(0);
      expect(rocCurve[i].tpr).toBeLessThanOrEqual(1);
    }
    expect(auc).toBeGreaterThanOrEqual(0);
    expect(auc).toBeLessThanOrEqual(1);
  });

  it('train with validationSplit and listen', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('_background_noise_');
    await transfer.collectExample('bar');
    await transfer.collectExample('bar');
    const history =
        await transfer.train({epochs: 3, batchSize: 2, validationSplit: 0.5}) as
        tfl.History;
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.acc.length).toEqual(3);
    expect(history.history.val_loss.length).toEqual(3);
    expect(history.history.val_acc.length).toEqual(3);

    // After the transfer learning is complete, listen() with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['_background_noise_', 'bar']);
    transfer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      transfer.stopListening().then(done);
    }, {probabilityThreshold: 0, invokeCallbackOnNoiseAndUnknown: true});
  });

  it('getMetadata works after transfer learning', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('_background_noise_');
    await transfer.collectExample('bar');
    await transfer.collectExample('bar');
    await transfer.train({epochs: 1, batchSize: 2, validationSplit: 0.5});

    const metadata = transfer.getMetadata();
    expect(metadata.tfjsSpeechCommandsVersion).toEqual(version);
    expect(metadata.modelName).toEqual('xfer1');
    expect(metadata.timeStamp != null).toEqual(true);
    expect(metadata.wordLabels).toEqual(['_background_noise_', 'bar']);
  });

  it('train with tf.data.Dataset, with fine-tuning', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('_background_noise_');
    await transfer.collectExample('_background_noise_');
    await transfer.collectExample('bar');
    await transfer.collectExample('bar');
    // Set the duration threshold to 0 to force using tf.data.Dataset
    // for training.
    const fitDatasetDurationMillisThreshold = 0;
    const [history, fineTuneHistory] = await transfer.train({
      epochs: 3,
      batchSize: 1,
      validationSplit: 0.5,
      fitDatasetDurationMillisThreshold,
      fineTuningEpochs: 2
    }) as [tfl.History, tfl.History];
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.acc.length).toEqual(3);
    expect(history.history.val_loss.length).toEqual(3);
    expect(history.history.val_acc.length).toEqual(3);
    expect(fineTuneHistory.history.loss.length).toEqual(2);
    expect(fineTuneHistory.history.acc.length).toEqual(2);
    expect(fineTuneHistory.history.val_loss.length).toEqual(2);
    expect(fineTuneHistory.history.val_acc.length).toEqual(2);
  });

  it('trainTransferLearningModel with fine-tuning + callback', async done => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');

    const oldKernel = secondLastBaseDenseLayer.getWeights()[0].dataSync();

    const historyObjects = await transfer.train({
      epochs: 3,
      batchSize: 2,
      fineTuningEpochs: 2,
      fineTuningOptimizer: 'adam'
    }) as tfl.History[];
    expect(historyObjects.length).toEqual(2);
    expect(historyObjects[0].history.loss.length).toEqual(3);
    expect(historyObjects[0].history.acc.length).toEqual(3);
    expect(historyObjects[1].history.loss.length).toEqual(2);
    expect(historyObjects[1].history.acc.length).toEqual(2);

    // Assert that the kernel has changed as a result of the fine-tuning.
    const newKernel = secondLastBaseDenseLayer.getWeights()[0].dataSync();

    let diffSumSquare = 0;
    for (let i = 0; i < newKernel.length; ++i) {
      const diff = newKernel[i] - oldKernel[i];
      diffSumSquare += diff * diff;
    }
    expect(diffSumSquare).toBeGreaterThan(1e-4);

    // After the transfer learning is complete, startStreaming with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);

    transfer.listen(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      transfer.stopListening().then(done);
    });
  });

  it('trainTransferLearningModel custom params and callback', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    for (let i = 0; i < 2; ++i) {
      await transfer.collectExample('bar');
    }
    const callbackEpochs: number[] = [];
    const history = await transfer.train({
      epochs: 5,
      callback: {
        onEpochEnd: async (epoch: number, logs: tfl.Logs) => {
          callbackEpochs.push(epoch);
        }
      }
    }) as tfl.History;
    expect(history.history.loss.length).toEqual(5);
    expect(history.history.acc.length).toEqual(5);
    expect(callbackEpochs).toEqual([0, 1, 2, 3, 4]);
  });

  it('trainTransferLearningModel fails without any examples', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    let errorCaught: Error;
    try {
      await transfer.train();
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toMatch(/no transfer learning example has been collected/);
  });

  it('trainTransferLearningModel fails with only 1 word', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    await transfer.collectExample('foo');
    let errorCaught: Error;
    try {
      await transfer.train();
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message).toMatch(/.*foo.*Requires at least 2/);
  });

  it('Invalid vocabulary name leads to Error', () => {
    expect(() => create('BROWSER_FFT', 'nonsensical_vocab'))
        .toThrowError(/Invalid vocabulary name.*\'nonsensical_vocab\'/);
  });

  it('getExamples()', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('bar');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    const barOut = transfer.getExamples('bar');
    expect(barOut.length).toEqual(2);
    expect(barOut[0].uid).toMatch(/^([0-9a-f]+\-)+[0-9a-f]+$/);
    expect(barOut[0].example.label).toEqual('bar');
    expect(barOut[1].uid).toMatch(/^([0-9a-f]+\-)+[0-9a-f]+$/);
    expect(barOut[1].example.label).toEqual('bar');
    const fooOut = transfer.getExamples('foo');
    expect(fooOut.length).toEqual(1);
    expect(fooOut[0].uid).toMatch(/^([0-9a-f]+\-)+[0-9a-f]+$/);
    expect(fooOut[0].example.label).toEqual('foo');
  });

  it('serializeExamples', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('bar');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    const artifacts =
        arrayBuffer2SerializedExamples(transfer.serializeExamples());

    // The examples are sorted alphabetically by their label.
    expect(artifacts.manifest).toEqual([
      {
        label: 'bar',
        spectrogramNumFrames: fakeNumFrames,
        spectrogramFrameSize: fakeColumnTruncateLength
      },
      {
        label: 'bar',
        spectrogramNumFrames: fakeNumFrames,
        spectrogramFrameSize: fakeColumnTruncateLength
      },
      {
        label: 'foo',
        spectrogramNumFrames: fakeNumFrames,
        spectrogramFrameSize: fakeColumnTruncateLength
      }
    ]);
    expect(artifacts.data.byteLength)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength * 4 * 3);
  });

  it('serializeExamples: limited word labels', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('bar');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    const artifacts =
        arrayBuffer2SerializedExamples(transfer.serializeExamples('bar'));

    // The examples are sorted alphabetically by their label.
    expect(artifacts.manifest).toEqual([
      {
        label: 'bar',
        spectrogramNumFrames: fakeNumFrames,
        spectrogramFrameSize: fakeColumnTruncateLength
      },
      {
        label: 'bar',
        spectrogramNumFrames: fakeNumFrames,
        spectrogramFrameSize: fakeColumnTruncateLength
      }
    ]);
    expect(artifacts.data.byteLength)
        .toEqual(fakeNumFrames * fakeColumnTruncateLength * 4 * 2);
  });

  it('removeExample & isDatasetEmpty', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    expect(transfer.isDatasetEmpty()).toEqual(true);
    await transfer.collectExample('bar');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    const fooExamples = transfer.getExamples('foo');
    transfer.removeExample(fooExamples[0].uid);
    expect(transfer.isDatasetEmpty()).toEqual(false);
    expect(transfer.countExamples()).toEqual({'bar': 2});
    expect(() => transfer.getExamples('foo'))
        .toThrowError('No example of label "foo" exists in dataset');
    const barExamples = transfer.getExamples('bar');
    transfer.removeExample(barExamples[0].uid);
    expect(transfer.isDatasetEmpty()).toEqual(false);
    expect(transfer.countExamples()).toEqual({'bar': 1});
    transfer.removeExample(barExamples[1].uid);
    expect(transfer.isDatasetEmpty()).toEqual(true);
  });

  it('serializeExamples fails on empty data', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    expect(() => transfer.serializeExamples()).toThrow();
  });

  it('loadExapmles, from empty state', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = base.createTransfer('xfer1');
    await transfer1.collectExample('foo');
    await transfer1.collectExample('bar');
    const transfer2 = base.createTransfer('xfer2');
    transfer2.loadExamples(transfer1.serializeExamples());

    expect(transfer2.countExamples()).toEqual({'bar': 1, 'foo': 1});

    // Assert that transfer2 can continue to collect new examples.
    await transfer2.collectExample('qux');
    expect(transfer2.countExamples()).toEqual({'bar': 1, 'foo': 1, 'qux': 1});
  });

  it('loadExapmles, from nonempty state, clearExisting = false', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = base.createTransfer('xfer1');
    await transfer1.collectExample('foo');
    await transfer1.collectExample('bar');
    const transfer2 = base.createTransfer('xfer2');
    await transfer2.collectExample('qux');
    transfer2.loadExamples(transfer1.serializeExamples());

    expect(transfer2.countExamples()).toEqual({'bar': 1, 'foo': 1, 'qux': 1});
  });

  it('loadExapmles, from nonempty state, clearExisting = true', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = base.createTransfer('xfer1');
    await transfer1.collectExample('foo');
    await transfer1.collectExample('bar');
    const transfer2 = base.createTransfer('xfer2');
    await transfer2.collectExample('qux');
    transfer2.loadExamples(transfer1.serializeExamples(), true);

    expect(transfer2.countExamples()).toEqual({'bar': 1, 'foo': 1});
  });

  it('loadExapmles, from a word-filtered dataset', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer1 = base.createTransfer('xfer1');
    await transfer1.collectExample('foo');
    await transfer1.collectExample('bar');
    const serialized = transfer1.serializeExamples('foo');
    const transfer2 = base.createTransfer('xfer2');
    transfer2.loadExamples(serialized);
    expect(transfer2.countExamples()).toEqual({'foo': 1});
    const examples = transfer2.getExamples('foo');
    expect(examples.length).toEqual(1);
    expect(examples[0].example.label).toEqual('foo');
  });

  it('collectExample with durationMultiplier = 1.5', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    const spectrogram =
        await transfer.collectExample('foo', {durationMultiplier: 1.5});
    expect(spectrogram.frameSize).toEqual(fakeColumnTruncateLength);
    const numFrames = spectrogram.data.length / fakeColumnTruncateLength;
    expect(numFrames).toEqual(fakeNumFrames * 1.5);
  });

  function setUpFakeLocalStorage(store: {[key: string]: string}) {
    // tslint:disable:no-any
    localStorageWrapper.localStorage = {
      getItem: (key: string) => {
        return store[key];
      },
      setItem: (key: string, value: string) => {
        store[key] = value;
      }
    } as any;
    // tslint:enable:no-any
  }

  function setUpFakeIndexedDB(artifactStore: tf.io.ModelArtifacts[]) {
    class FakeIndexedDBHandler implements tf.io.IOHandler {
      constructor(readonly artifactStore: tf.io.ModelArtifacts[]) {}

      async save(artifacts: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> {
        this.artifactStore.push(artifacts);
        return null;
      }

      async load(): Promise<tf.io.ModelArtifacts> {
        return this.artifactStore[this.artifactStore.length - 1];
      }
    }

    const handler = new FakeIndexedDBHandler(artifactStore);
    function fakeIndexedDBRouter(url: string|string[]): tf.io.IOHandler {
      if (!Array.isArray(url) && url.startsWith('indexeddb://')) {
        return handler;
      } else {
        return null;
      }
    }
    tf.io.registerSaveRouter(fakeIndexedDBRouter);
    tf.io.registerLoadRouter(fakeIndexedDBRouter);
  }

  it('Save and load transfer model via indexeddb://', async () => {
    setUpFakes();
    const localStore: {[key: string]: string} = {};
    setUpFakeLocalStorage(localStore);
    const indexedDBStore: tf.io.ModelArtifacts[] = [];
    setUpFakeIndexedDB(indexedDBStore);

    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    await transfer.train({epochs: 1});

    const xs = tf.ones([1, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const out0 = await transfer.recognize(xs);

    await transfer.save();

    const savedMetadata = JSON.parse(localStore[SAVED_MODEL_METADATA_KEY]);
    expect(savedMetadata['xfer1']['modelName']).toEqual('xfer1');
    expect(savedMetadata['xfer1']['wordLabels']).toEqual(['bar', 'foo']);
    expect(indexedDBStore.length).toEqual(1);
    const modelPrime =
        await tfl.models.modelFromJSON(indexedDBStore[0].modelTopology as {});
    expect(modelPrime.layers.length).toEqual(4);
    expect(indexedDBStore[0].weightSpecs.length).toEqual(4);

    // Load the transfer model back.
    const base2 = new BrowserFftSpeechCommandRecognizer();
    await base2.ensureModelLoaded();
    // Disable the spy on tf.loadLayersModel() first, so the subsequent
    // tf.loadLayersModel() call during the load() call can use the fake
    // IndexedDB handler created above.
    tfLoadModelSpy.and.callThrough();
    const transfer2 = base2.createTransfer('xfer1');
    await transfer2.load();
    expect(transfer2.wordLabels()).toEqual(['bar', 'foo']);
    const out1 = await transfer2.recognize(xs);
    // The new prediction scores from the loaded transfer model should match
    // the prediction scores from the original transfer model.
    expect(out1.scores).toEqual(out0.scores);
  });

  it('Save model via custom file:// route', async () => {
    setUpFakes();

    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    const transfer = base.createTransfer('xfer1');
    await transfer.collectExample('foo');
    await transfer.collectExample('bar');
    await transfer.train({epochs: 1});

    const tempSavePath = tempfile();
    await transfer.save(`file://${tempSavePath}`);

    // Disable the spy on tf.loadLayersModel() first, so the subsequent
    // tf.loadLayersModel() call during the load() call can use the fake
    // IndexedDB handler created above.
    tfLoadModelSpy.and.callThrough();
    const modelPrime =
        await tfl.loadLayersModel(`file://${tempSavePath}/model.json`);
    expect(modelPrime.outputs.length).toEqual(1);
    expect(modelPrime.outputs[0].shape).toEqual([null, 2]);

    rimraf(tempSavePath, () => {});
  });

  it('listSavedTransferModels', async () => {
    spyOn(tf.io, 'listModels').and.callFake(() => {
      return {
        'indexeddb://tfjs-speech-commands-model/model1':
            {'dateSaved': '2018-12-06T04:25:08.153Z'}
      };
    });
    expect(await listSavedTransferModels()).toEqual(['model1']);
  });

  it('deleteSavedTransferModel', async () => {
    const localStore: {[key: string]: string} = {
      'tfjs-speech-commands-saved-model-metadata':
          JSON.stringify({'foo': {'wordLabels': ['a', 'b']}})
    };
    setUpFakeLocalStorage(localStore);
    const removedModelPaths: string[] = [];
    spyOn(tf.io, 'removeModel').and.callFake((modelPath: string) => {
      removedModelPaths.push(modelPath);
    });
    await deleteSavedTransferModel('foo');
    expect(removedModelPaths).toEqual([
      'indexeddb://tfjs-speech-commands-model/foo'
    ]);
    expect(localStore).toEqual({
      'tfjs-speech-commands-saved-model-metadata': '{}'
    });
  });
});
