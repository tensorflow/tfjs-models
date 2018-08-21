/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {BrowserFftSpeechCommandRecognizer} from './browser_fft_recognizer';
import * as BrowserFftUtils from './browser_fft_utils';
import {FakeAudioContext, FakeAudioMediaStream} from './browser_test_utils';
import {SpeechCommandRecognizerResult} from './types';

describeWithFlags('Browser FFT recognizer', tf.test_util.NODE_ENVS, () => {
  const fakeWords: string[] = [
    '_background_noise_', 'down', 'eight', 'five', 'four', 'go', 'left', 'nine',
    'one', 'right', 'seven', 'six', 'stop', 'three', 'two', 'up', 'zero'
  ];
  const fakeWordsNoiseAndUnknownOnly: string[] =
      ['_background_noise_', '_unknown_'];

  const fakeNumFrames = 42;
  const fakeColumnTruncateLength = 232;

  function setUpFakes(model?: tf.Sequential, backgroundAndNoiseOnly = false) {
    const words =
        backgroundAndNoiseOnly ? fakeWordsNoiseAndUnknownOnly : fakeWords;
    const numWords = words.length;
    spyOn(tf, 'loadModel').and.callFake((url: string) => {
      if (model == null) {
        model = tf.sequential();
        model.add(tf.layers.flatten(
            {inputShape: [fakeNumFrames, fakeColumnTruncateLength, 1]}));
        model.add(tf.layers.dense(
            {units: numWords, useBias: false, activation: 'softmax'}));
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
    expect(recognizer.isStreaming()).toEqual(false);
    expect(recognizer.params().sampleRateHz).toEqual(44100);
    expect(recognizer.params().fftSize).toEqual(1024);
    expect(recognizer.params().columnBufferLength).toEqual(1024);
  });

  it('ensureModelLoaded succeeds', async () => {
    setUpFakes();

    const recognizer = new BrowserFftSpeechCommandRecognizer();
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual(fakeWords);
    expect(recognizer.params().spectrogramDurationMillis)
        .toBeCloseTo(fakeNumFrames * 1024 / 44100 * 1e3);
    expect(recognizer.model instanceof tf.Model).toEqual(true);
    expect(recognizer.modelInputShape()).toEqual([
      null, fakeNumFrames, fakeColumnTruncateLength, 1
    ]);
  });

  it('ensureModelLoaded fails: words - model output mismatch', async () => {
    const fakeModel = tf.sequential();
    fakeModel.add(tf.layers.flatten(
        {inputShape: [fakeNumFrames, fakeColumnTruncateLength, 1]}));
    fakeModel.add(tf.layers.dense({units: 12, activation: 'softmax'}));
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

  it('startStreaming call with invalid overlapFactor', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;

    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: -1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);

    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: 1});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);

    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {},
          {overlapFactor: 1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message).toMatch(/Expected overlapFactor/);
  });

  it('startStreaming call with invalid probabilityThreshold', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    let caughtError: Error;
    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {},
          {probabilityThreshold: 1.2});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Invalid probabilityThreshold value: 1\.2/);

    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {},
          {probabilityThreshold: -0.1});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Invalid probabilityThreshold value: -0\.1/);
  });

  it('streaming: overlapFactor = 0', async done => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    const numCallbacksToComplete = 2;
    let numCallbacksCompleted = 0;
    const tensorCounts: number[] = [];
    const callbackTimestamps: number[] = [];
    recognizer.startStreaming(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(fakeWords.length);

      callbackTimestamps.push(tf.util.now());
      if (callbackTimestamps.length > 1) {
        expect(
            callbackTimestamps[callbackTimestamps.length - 1] -
            callbackTimestamps[callbackTimestamps.length - 2])
            .toBeGreaterThanOrEqual(
                recognizer.params().spectrogramDurationMillis);
      }

      tensorCounts.push(tf.memory().numTensors);
      if (tensorCounts.length > 1) {
        // Assert no memory leak.
        expect(tensorCounts[tensorCounts.length - 1])
            .toEqual(tensorCounts[tensorCounts.length - 2]);
      }

      // spectrogram is not provided by default.
      expect(result.spectrogram).toBeUndefined();

      if (++numCallbacksCompleted >= numCallbacksToComplete) {
        recognizer.stopStreaming().then(done);
      }
    }, {overlapFactor: 0, invokeCallbackOnNoiseAndUnknown: true});
  });

  it('streaming: overlapFactor = 0.5, includeSpectrogram', async done => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    const numCallbacksToComplete = 2;
    let numCallbacksCompleted = 0;
    const tensorCounts: number[] = [];
    const callbackTimestamps: number[] = [];
    await recognizer.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {
          expect((result.scores as Float32Array).length)
              .toEqual(fakeWords.length);

          callbackTimestamps.push(tf.util.now());
          if (callbackTimestamps.length > 1) {
            expect(
                callbackTimestamps[callbackTimestamps.length - 1] -
                callbackTimestamps[callbackTimestamps.length - 2])
                .toBeGreaterThanOrEqual(
                    recognizer.params().spectrogramDurationMillis * 0.5);
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
            recognizer.stopStreaming().then(done);
          }
        },
        {
          overlapFactor: 0.5,
          includeSpectrogram: true,
          invokeCallbackOnNoiseAndUnknown: true
        });
  });

  it('streaming: invokeCallbackOnNoiseAndUnknown = false', async done => {
    setUpFakes(null, true);
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    let callbackInvokeCount = 0;
    await recognizer.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {
          callbackInvokeCount++;
        },
        {overlapFactor: 0.5, invokeCallbackOnNoiseAndUnknown: false});

    setTimeout(() => {
      recognizer.stopStreaming();
      // Due to `invokeCallbackOnNoiseAndUnknown: false` and the fact that the
      // vocabulary contains only _background_noise_ and _unknown_, the callback
      // should have never been called.
      expect(callbackInvokeCount).toEqual(0);
      done();
    }, 1000);
  });

  it('streaming: invokeCallbackOnNoiseAndUnknown = true', async done => {
    setUpFakes(null, true);
    const recognizer = new BrowserFftSpeechCommandRecognizer();

    let callbackInvokeCount = 0;
    await recognizer.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {
          callbackInvokeCount++;
        },
        {overlapFactor: 0.5, invokeCallbackOnNoiseAndUnknown: true});

    setTimeout(() => {
      recognizer.stopStreaming();
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
    await recognizer.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {});
    expect(recognizer.isStreaming()).toEqual(true);

    let caughtError: Error;
    try {
      await recognizer.startStreaming(
          async (result: SpeechCommandRecognizerResult) => {});
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toEqual('Cannot start streaming again when streaming is ongoing.');
    expect(recognizer.isStreaming()).toEqual(true);

    await recognizer.stopStreaming();
    expect(recognizer.isStreaming()).toEqual(false);
  });

  it('Attempt to stop streaming twice leads to Error', async () => {
    setUpFakes();
    const recognizer = new BrowserFftSpeechCommandRecognizer();
    await recognizer.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {});
    expect(recognizer.isStreaming()).toEqual(true);

    await recognizer.stopStreaming();
    expect(recognizer.isStreaming()).toEqual(false);

    let caughtError: Error;
    try {
      await recognizer.stopStreaming();
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toEqual('Cannot stop streaming when streaming is not ongoing.');
    expect(recognizer.isStreaming()).toEqual(false);
  });
});
