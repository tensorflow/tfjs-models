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

import {SpeechCommandBrowserFftRecognizer} from './browser_fft_recognizer';
import * as BrowserFftUtils from './browser_fft_utils';

describeWithFlags('Browser FFT recognizer', tf.test_util.NODE_ENVS, () => {
  const fakeWords: string[] = [
    '_background_noise_', 'down', 'eight', 'five', 'four', 'go', 'left', 'nine',
    'one', 'right', 'seven', 'six', 'stop', 'three', 'two', 'up', 'zero'
  ];

  function setUpFakes(model?: tf.Sequential) {
    spyOn(tf, 'loadModel').and.callFake((url: string) => {
      if (model == null) {
        model = tf.sequential();
        model.add(tf.layers.flatten({inputShape: [42, 232, 1]}));
        model.add(tf.layers.dense({units: 17, activation: 'softmax'}));
      }
      return model;
    });
    spyOn(BrowserFftUtils, 'loadMetadataJson')
        .and.callFake(async (url: string) => {
          return {words: fakeWords};
        });
  }

  it('Constructor', () => {
    const recognizer = new SpeechCommandBrowserFftRecognizer();
    expect(recognizer.isStreaming()).toEqual(false);
    expect(recognizer.params().sampleRateHz).toEqual(44100);
    expect(recognizer.params().fftSize).toEqual(1024);
    expect(recognizer.params().columnBufferLength).toEqual(1024);
    expect(recognizer.params().columnHopLength).toEqual(1024);
  });

  it('ensureModelLoaded succeeds', async () => {
    setUpFakes();

    const recognizer = new SpeechCommandBrowserFftRecognizer();
    await recognizer.ensureModelLoaded();
    expect(recognizer.wordLabels()).toEqual(fakeWords);
    expect(recognizer.params().spectrogramDurationMillis)
        .toBeCloseTo(42 * 1024 / 44100 * 1e3);
    expect(recognizer.model instanceof tf.Model).toEqual(true);
  });

  it('ensureModelLoaded fails: words - model output mismatch', async () => {
    const fakeModel = tf.sequential();
    fakeModel.add(tf.layers.flatten({inputShape: [42, 232, 1]}));
    fakeModel.add(tf.layers.dense({units: 12, activation: 'softmax'}));
    setUpFakes(fakeModel);

    const recognizer = new SpeechCommandBrowserFftRecognizer();
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

    const spectrogram = tf.zeros([1, 42, 232, 1]);
    const recognizer = new SpeechCommandBrowserFftRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(output.scores instanceof Float32Array).toEqual(true);
    expect(output.scores.length).toEqual(17);
  });

  it('Offline recognize succeeds with batched tf.Tensor', async () => {
    setUpFakes();

    // TODO(cais):
    const spectrogram = tf.zeros([3, 42, 232, 1]);
    const recognizer = new SpeechCommandBrowserFftRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(Array.isArray(output.scores)).toEqual(true);
    expect(output.scores.length).toEqual(3);
    for (let i = 0; i < 3; ++i) {
      expect((output.scores[i] as Float32Array).length).toEqual(17);
    }
  });

  it('Offline recognize fails due to incorrect shape', async () => {
    setUpFakes();

    const spectrogram = tf.zeros([1, 42, 232, 2]);
    const recognizer = new SpeechCommandBrowserFftRecognizer();
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

    const spectrogram = new Float32Array(42 * 232 * 1);
    const recognizer = new SpeechCommandBrowserFftRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(output.scores instanceof Float32Array).toEqual(true);
    expect(output.scores.length).toEqual(17);
  });

  it('Offline recognize succeeds with batched Float32Array', async () => {
    setUpFakes();

    const spectrogram = new Float32Array(2 * 42 * 232 * 1);
    const recognizer = new SpeechCommandBrowserFftRecognizer();
    const output = await recognizer.recognize(spectrogram);
    expect(Array.isArray(output.scores)).toEqual(true);
    expect(output.scores.length).toEqual(2);
    for (let i = 0; i < 2; ++i) {
      expect((output.scores[i] as Float32Array).length).toEqual(17);
    }
  });
});
