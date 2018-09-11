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
import {create} from './index';
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
        model.add(tf.layers.dense({units: 4, activation: 'relu'}));
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

  it('collectTransferLearningExample default transerf model', async () => {
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

  it('Concurrent collectExample+startStreaming fails', async () => {
    setUpFakes();
    const base = new BrowserFftSpeechCommandRecognizer();
    await base.ensureModelLoaded();
    await base.startStreaming(
        async (result: SpeechCommandRecognizerResult) => {});
    expect(base.isStreaming()).toEqual(true);

    const transfer = base.createTransfer('xfer1');
    let caughtError: Error;
    try {
      await transfer.collectExample('foo');
    } catch (err) {
      caughtError = err;
    }
    expect(caughtError.message)
        .toMatch(/Cannot start collection of transfer-learning example/);
    expect(base.isStreaming()).toEqual(true);

    await base.stopStreaming();
    expect(base.isStreaming()).toEqual(false);
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
    const transferHead = (transfer as any).transferHead as tf.Sequential;
    const numLayers = transferHead.layers.length;
    const oldTransferKernel =
        transferHead.getLayer(null, numLayers - 1).getWeights()[0].dataSync();

    const history = await transfer.train({optimizer: tf.train.sgd(1)});
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
    const newTransferKernel =
        transferHead.getLayer(null, numLayers - 1).getWeights()[0].dataSync();
    baseModelOldWeightValues.forEach((oldWeight, i) => {
      tf.test_util.expectArraysClose(baseModelNewWeightValues[i], oldWeight);
    });
    // Verify that the weight of the transfer-learning head model changes
    // after training.
    expect(tf.tensor1d(newTransferKernel)
               .sub(tf.tensor1d(oldTransferKernel))
               .abs()
               .max()
               .dataSync()[0])
        .toBeGreaterThan(1e-3);

    // Test recognize() with the transfer recognizer.
    const spectrogram =
        tf.zeros([1, fakeNumFrames, fakeColumnTruncateLength, 1]);
    const result = await transfer.recognize(spectrogram);
    expect(result.scores.length).toEqual(2);

    // After the transfer learning is complete, startStreaming with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);
    transfer.startStreaming(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      transfer.stopStreaming().then(done);
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
    const history = await transfer.train({epochs: 10, batchSize: 2});
    expect(history.history.loss.length).toEqual(10);
    expect(history.history.acc.length).toEqual(10);

    // After the transfer learning is complete, startStreaming with the
    // transfer-learned model's name should give scores only for the
    // transfer-learned model.
    expect(base.wordLabels()).toEqual(fakeWords);
    expect(transfer.wordLabels()).toEqual(['bar', 'foo']);
    transfer.startStreaming(async (result: SpeechCommandRecognizerResult) => {
      expect((result.scores as Float32Array).length).toEqual(2);
      transfer.stopStreaming().then(done);
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
        onEpochEnd: async (epoch, logs) => {
          callbackEpochs.push(epoch);
        }
      }
    });
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

  // TODO(cais): Add tests for saving and loading of transfer-learned models.
});
