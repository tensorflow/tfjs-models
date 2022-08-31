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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {normalize} from './browser_fft_utils';
import {arrayBuffer2SerializedExamples, BACKGROUND_NOISE_TAG, Dataset, DATASET_SERIALIZATION_DESCRIPTOR, DATASET_SERIALIZATION_VERSION, deserializeExample, getMaxIntensityFrameIndex, getValidWindows, serializeExample, spectrogram2IntensityCurve, SpectrogramAndTargetsTfDataset} from './dataset';
import {string2ArrayBuffer} from './generic_utils';
import {expectTensorsClose} from './test_utils';
import {Example, RawAudioData, SpectrogramData} from './types';

describe('Dataset', () => {
  const FAKE_NUM_FRAMES = 4;
  const FAKE_FRAME_SIZE = 16;

  function getFakeExample(
      label: string, numFrames = FAKE_NUM_FRAMES, frameSize = FAKE_FRAME_SIZE,
      spectrogramData?: number[]): Example {
    if (spectrogramData == null) {
      spectrogramData = [];
      let counter = 0;
      for (let i = 0; i < numFrames * frameSize; ++i) {
        spectrogramData.push(counter++);
      }
    }
    return {
      label,
      spectrogram: {data: new Float32Array(spectrogramData), frameSize}
    };
  }

  function addThreeExamplesToDataset(
      dataset: Dataset, labelA = 'a', labelB = 'b'): string[] {
    const ex1 = getFakeExample(labelA);
    const uid1 = dataset.addExample(ex1);
    const ex2 = getFakeExample(labelA);
    const uid2 = dataset.addExample(ex2);
    const ex3 = getFakeExample(labelB);
    const uid3 = dataset.addExample(ex3);
    return [uid1, uid2, uid3];
  }

  it('Constructor', () => {
    const dataset = new Dataset();
    expect(dataset.empty()).toEqual(true);
    expect(dataset.size()).toEqual(0);
  });

  it('addExample', () => {
    const dataset = new Dataset();

    const uids: string[] = [];
    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(uid1).toMatch(/^([0-9a-f]+\-)+[0-9a-f]+$/);
    uids.push(uid1);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(1);
    expect(dataset.getExampleCounts()).toEqual({'a': 1});

    const ex2 = getFakeExample('a');
    const uid2 = dataset.addExample(ex2);
    expect(uids.indexOf(uid2)).toEqual(-1);
    uids.push(uid2);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(2);
    expect(dataset.getExampleCounts()).toEqual({'a': 2});

    const ex3 = getFakeExample('b');
    const uid3 = dataset.addExample(ex3);
    expect(uids.indexOf(uid3)).toEqual(-1);
    uids.push(uid3);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(3);
    expect(dataset.getExampleCounts()).toEqual({'a': 2, 'b': 1});
  });

  it('addExample with null fails', () => {
    const dataset = new Dataset();
    expect(() => dataset.addExample(null))
        .toThrowError('Got null or undefined example');
  });

  it('addExample with invalid label fails', () => {
    const dataset = new Dataset();
    expect(() => dataset.addExample(getFakeExample(null)))
        .toThrowError(/Expected label to be a non-empty string.*null/);
    expect(() => dataset.addExample(getFakeExample(undefined)))
        .toThrowError(/Expected label to be a non-empty string.*undefined/);
    expect(() => dataset.addExample(getFakeExample('')))
        .toThrowError(/Expected label to be a non-empty string/);
  });

  it('durationMillis', () => {
    const dataset = new Dataset();
    expect(dataset.durationMillis()).toEqual(0);
    const ex1 = getFakeExample('a');
    dataset.addExample(ex1);
    const durationPerExample = dataset.durationMillis();
    expect(durationPerExample).toBeGreaterThan(0);
    const ex2 = getFakeExample('b');
    dataset.addExample(ex2);
    expect(dataset.durationMillis()).toEqual(durationPerExample * 2);
    dataset.clear();
    expect(dataset.durationMillis()).toEqual(0);
  });

  it('merge two non-empty datasets', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);
    const datasetPrime = new Dataset();
    addThreeExamplesToDataset(datasetPrime);
    const ex = getFakeExample('foo');
    datasetPrime.addExample(ex);
    const duration0 = dataset.durationMillis();
    dataset.merge(datasetPrime);
    expect(dataset.getExampleCounts()).toEqual({a: 4, b: 2, foo: 1});
    // Check that the content of the incoming dataset is not affected.
    expect(datasetPrime.getExampleCounts()).toEqual({a: 2, b: 1, foo: 1});
    expect(dataset.durationMillis())
        .toEqual(duration0 + datasetPrime.durationMillis());
  });

  it('merge non-empty dataset into an empty one', () => {
    const dataset = new Dataset();
    const datasetPrime = new Dataset();
    addThreeExamplesToDataset(datasetPrime);
    dataset.merge(datasetPrime);
    expect(dataset.getExampleCounts()).toEqual({a: 2, b: 1});
    // Check that the content of the incoming dataset is not affected.
    expect(datasetPrime.getExampleCounts()).toEqual({a: 2, b: 1});
  });

  it('merge empty dataset into an non-empty one', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);
    const datasetPrime = new Dataset();
    dataset.merge(datasetPrime);
    expect(dataset.getExampleCounts()).toEqual({a: 2, b: 1});
    // Check that the content of the incoming dataset is not affected.
    expect(datasetPrime.empty()).toEqual(true);
  });

  it('getExamples', () => {
    const dataset = new Dataset();
    const [uid1, uid2, uid3] = addThreeExamplesToDataset(dataset);
    const out1 = dataset.getExamples('a');
    expect(out1.length).toEqual(2);
    expect(out1[0].uid).toEqual(uid1);
    expect(out1[0].example.label).toEqual('a');
    expect(out1[1].uid).toEqual(uid2);
    expect(out1[1].example.label).toEqual('a');
    const out2 = dataset.getExamples('b');
    expect(out2.length).toEqual(1);
    expect(out2[0].uid).toEqual(uid3);
    expect(out2[0].example.label).toEqual('b');
  });

  it('getExamples after addExample', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);
    const out1 = dataset.getExamples('a');
    expect(out1.length).toEqual(2);
    expect(out1[0].uid).toEqual(uid1);
    expect(out1[0].example.label).toEqual('a');
    expect(out1[1].uid).toEqual(uid2);
    expect(out1[1].example.label).toEqual('a');

    const ex = getFakeExample('a');
    const uid4 = dataset.addExample(ex);
    const out2 = dataset.getExamples('a');
    expect(out2.length).toEqual(3);
    expect(out2[0].uid).toEqual(uid1);
    expect(out2[0].example.label).toEqual('a');
    expect(out2[1].uid).toEqual(uid2);
    expect(out2[1].example.label).toEqual('a');
    expect(out2[2].uid).toEqual(uid4);
    expect(out2[2].example.label).toEqual('a');
  });

  it('getExamples after removeExample', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);
    const out1 = dataset.getExamples('a');
    expect(out1.length).toEqual(2);
    expect(out1[0].uid).toEqual(uid1);
    expect(out1[0].example.label).toEqual('a');
    expect(out1[1].uid).toEqual(uid2);
    expect(out1[1].example.label).toEqual('a');

    dataset.removeExample(uid1);
    const out2 = dataset.getExamples('a');
    expect(out2.length).toEqual(1);
    expect(out2[0].uid).toEqual(uid2);
    expect(out2[0].example.label).toEqual('a');

    dataset.removeExample(uid2);
    expect(() => dataset.getExamples('a'))
        .toThrowError(/No example .*a.* exists/);
  });

  it('getExamples after removeExample followed by addExample', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);
    const out1 = dataset.getExamples('a');
    expect(out1.length).toEqual(2);
    expect(out1[0].uid).toEqual(uid1);
    expect(out1[0].example.label).toEqual('a');
    expect(out1[1].uid).toEqual(uid2);
    expect(out1[1].example.label).toEqual('a');

    dataset.removeExample(uid1);
    const out2 = dataset.getExamples('a');
    expect(out2.length).toEqual(1);
    expect(out2[0].uid).toEqual(uid2);
    expect(out2[0].example.label).toEqual('a');

    const ex = getFakeExample('a');
    const uid4 = dataset.addExample(ex);
    const out3 = dataset.getExamples('a');
    expect(out3.length).toEqual(2);
    expect(out3[0].uid).toEqual(uid2);
    expect(out3[0].example.label).toEqual('a');
    expect(out3[1].uid).toEqual(uid4);
    expect(out3[1].example.label).toEqual('a');
  });

  it('getExamples with nonexistent label fails', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);
    expect(() => dataset.getExamples('labelC'))
        .toThrowError(/No example .*labelC.* exists/);
  });

  it('removeExample', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(1);
    expect(dataset.getExampleCounts()).toEqual({'a': 1});

    const ex2 = getFakeExample('a');
    const uid2 = dataset.addExample(ex2);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(2);
    expect(dataset.getExampleCounts()).toEqual({'a': 2});

    const ex3 = getFakeExample('b');
    const uid3 = dataset.addExample(ex3);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(3);
    expect(dataset.getExampleCounts()).toEqual({'a': 2, 'b': 1});

    dataset.removeExample(uid1);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(2);
    expect(dataset.getExampleCounts()).toEqual({'a': 1, 'b': 1});

    dataset.removeExample(uid2);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(1);
    expect(dataset.getExampleCounts()).toEqual({'b': 1});

    dataset.removeExample(uid3);
    expect(dataset.empty()).toEqual(true);
    expect(dataset.size()).toEqual(0);
    expect(dataset.getExampleCounts()).toEqual({});
  });

  it('removeExample with nonexistent UID fails', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    dataset.removeExample(uid1);
    expect(() => dataset.removeExample(uid1))
        .toThrowError(/Nonexistent example UID/);
  });

  it('setExampleKeyFrameIndex: works`', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    dataset.setExampleKeyFrameIndex(uid1, 1);
    expect(ex1.spectrogram.keyFrameIndex).toEqual(1);
    const numFrames = ex1.spectrogram.data.length / ex1.spectrogram.frameSize;
    dataset.setExampleKeyFrameIndex(uid1, numFrames - 1);
    expect(ex1.spectrogram.keyFrameIndex).toEqual(numFrames - 1);
  });

  it('setExampleFrameIndex: serialization-deserialization', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    const numFrames = ex1.spectrogram.data.length / ex1.spectrogram.frameSize;
    dataset.setExampleKeyFrameIndex(uid1, numFrames - 1);
    expect(ex1.spectrogram.keyFrameIndex).toEqual(numFrames - 1);

    const datasetPrime = new Dataset(dataset.serialize());
    const ex1Prime = datasetPrime.getExamples('a')[0];
    expect(ex1Prime.example.spectrogram.keyFrameIndex).toEqual(numFrames - 1);
  });

  it('setExampleKeyFrameIndex: Invalid example UID leads to error`', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(() => dataset.setExampleKeyFrameIndex(uid1 + '_foo', 0))
        .toThrowError(/Nonexistent example UID/);
  });

  it('setExampleKeyFrameIndex: Negative index leads to error`', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(() => dataset.setExampleKeyFrameIndex(uid1, -1))
        .toThrowError(/Invalid keyFrameIndex/);
  });

  it('setExampleKeyFrameIndex: Too large index leads to error`', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const numFrames = ex1.spectrogram.data.length / ex1.spectrogram.frameSize;
    const uid1 = dataset.addExample(ex1);
    expect(() => dataset.setExampleKeyFrameIndex(uid1, numFrames))
        .toThrowError(/Invalid keyFrameIndex/);
    expect(() => dataset.setExampleKeyFrameIndex(uid1, numFrames + 1))
        .toThrowError(/Invalid keyFrameIndex/);
  });

  it('setExampleKeyFrameIndex: Non-integr value leads to error`', () => {
    const dataset = new Dataset();

    const ex1 = getFakeExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(() => dataset.setExampleKeyFrameIndex(uid1, 0.5))
        .toThrowError(/Invalid keyFrameIndex/);
  });

  it('getVocabulary', () => {
    const dataset = new Dataset();
    expect(dataset.getVocabulary()).toEqual([]);

    const ex1 = getFakeExample('a');
    const ex2 = getFakeExample('a');
    const ex3 = getFakeExample('b');

    const uid1 = dataset.addExample(ex1);
    expect(dataset.getVocabulary()).toEqual(['a']);
    const uid2 = dataset.addExample(ex2);
    expect(dataset.getVocabulary()).toEqual(['a']);
    const uid3 = dataset.addExample(ex3);
    expect(dataset.getVocabulary()).toEqual(['a', 'b']);

    dataset.removeExample(uid1);
    expect(dataset.getVocabulary()).toEqual(['a', 'b']);
    dataset.removeExample(uid2);
    expect(dataset.getVocabulary()).toEqual(['b']);
    dataset.removeExample(uid3);
    expect(dataset.getVocabulary()).toEqual([]);
  });

  it('getSpectrogramsAsTensors with label', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    const out1 = dataset.getData('a') as {xs: tf.Tensor, ys: tf.Tensor};
    expect(out1.xs.shape).toEqual([2, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
    expect(out1.ys).toBeUndefined();
    const out2 = dataset.getData('b') as {xs: tf.Tensor, ys: tf.Tensor};
    expect(out2.xs.shape).toEqual([1, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
    expect(out2.ys).toBeUndefined();
  });

  it('getSpectrogramsAsTensors after removeExample', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);

    dataset.removeExample(uid1);
    const out1 = dataset.getData(null, {shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(out1.xs.shape).toEqual([2, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
    expectTensorsClose(out1.ys, tf.tensor2d([[1, 0], [0, 1]]));

    const out2 = dataset.getData('a') as {xs: tf.Tensor, ys: tf.Tensor};
    expect(out2.xs.shape).toEqual([1, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);

    dataset.removeExample(uid2);
    expect(() => dataset.getData('a'))
        .toThrowError(/Label a is not in the vocabulary/);

    const out3 = dataset.getData('b') as {xs: tf.Tensor, ys: tf.Tensor};
    expect(out3.xs.shape).toEqual([1, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
  });

  it('getSpectrogramsAsTensors w/o label on one-word vocabulary fails', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);
    dataset.removeExample(uid1);
    dataset.removeExample(uid2);

    expect(() => dataset.getData())
        .toThrowError(/requires .* at least two words/);
  });

  it('getSpectrogramsAsTensors without label', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    const out = dataset.getData(null, {shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(out.xs.shape).toEqual([3, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
    expectTensorsClose(out.ys, tf.tensor2d([[1, 0], [1, 0], [0, 1]]));
  });

  it('getSpectrogramsAsTensors without label as tf.data.Dataset', async () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    const [trainDataset, valDataset] = dataset.getData(null, {
      getDataset: true,
      datasetBatchSize: 1,
      datasetValidationSplit: 1 / 3
    }) as [SpectrogramAndTargetsTfDataset, SpectrogramAndTargetsTfDataset];
    let numTrain = 0;
    await trainDataset.forEachAsync(
        (xAndY: {xs: tf.Tensor2D, ys: tf.Tensor2D}) => {
          const {xs, ys} = xAndY;
          numTrain++;
          expect(xs.shape).toEqual([1, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
          expect(xs.isDisposed).toEqual(false);
          expect(ys.shape).toEqual([1, 2]);
          expect(ys.isDisposed).toEqual(false);
        });
    expect(numTrain).toEqual(2);
    let numVal = 0;
    await valDataset.forEachAsync(
        (xAndY: {xs: tf.Tensor2D, ys: tf.Tensor2D}) => {
          const {xs, ys} = xAndY;
          numVal++;
          expect(xs.shape).toEqual([1, FAKE_NUM_FRAMES, FAKE_FRAME_SIZE, 1]);
          expect(xs.isDisposed).toEqual(false);
          expect(ys.shape).toEqual([1, 2]);
          expect(ys.isDisposed).toEqual(false);
        });
    expect(numVal).toEqual(1);
  });

  it('getData w/ mixing-noise augmentation: get tf.data.Dataset', async () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        BACKGROUND_NOISE_TAG, 6, 2,
        [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const numFrames = 3;
    const [trainDataset, valDataset] = dataset.getData(null, {
      numFrames,
      hopFrames: 1,
      augmentByMixingNoiseRatio: 0.5,
      getDataset: true,
      datasetBatchSize: 1,
      datasetValidationSplit: 1 / 3
    }) as [SpectrogramAndTargetsTfDataset, SpectrogramAndTargetsTfDataset];

    let numTrain = 0;
    await trainDataset.forEachAsync(
        (xAndY: {xs: tf.Tensor2D, ys: tf.Tensor2D}) => {
          const {xs, ys} = xAndY;
          numTrain++;
          expect(xs.shape).toEqual([1, numFrames, 2, 1]);
          expect(xs.isDisposed).toEqual(false);
          expect(ys.shape).toEqual([1, 2]);
          expect(ys.isDisposed).toEqual(false);
        });
    let numVal = 0;
    await valDataset.forEachAsync(
        (xAndY: {xs: tf.Tensor2D, ys: tf.Tensor2D}) => {
          const {xs, ys} = xAndY;
          numVal++;
          expect(xs.shape).toEqual([1, numFrames, 2, 1]);
          expect(xs.isDisposed).toEqual(false);
          expect(ys.shape).toEqual([1, 2]);
          expect(ys.isDisposed).toEqual(false);
        });
    expect(numTrain).toEqual(7);  // Without augmentation, it'd be 5.
    expect(numVal).toEqual(3);    // Without augmentation, it'd be 2.
  });

  it('getData w/ mixing-noise augmentation w/o noise tag errors', async () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));
    // Lacks BACKGROUND_NOISE_TAG.

    const numFrames = 3;
    expect(() => dataset.getData(null, {
      numFrames,
      hopFrames: 1,
      augmentByMixingNoiseRatio: 0.5,
      getDataset: true,
      datasetBatchSize: 1,
      datasetValidationSplit: 1 / 3
    })).toThrowError(/Cannot perform augmentation .* no example .*noise/);
  });

  it('getSpectrogramsAsTensors with invalid valSplit leads to error', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);
    expect(() => dataset.getData(null, {
      getDataset: true,
      datasetValidationSplit: 1.2
    })).toThrowError(/Invalid dataset validation split/);
  });

  it('getSpectrogramsAsTensors on nonexistent label fails', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    expect(() => dataset.getData('label3'))
        .toThrowError(/Label label3 is not in the vocabulary/);
  });

  it('getSpectrogramsAsTensors on empty Dataset fails', () => {
    const dataset = new Dataset();
    expect(() => dataset.getData())
        .toThrowError(/Cannot get spectrograms as tensors because.*empty/);
  });

  it('Ragged example lengths and one window per example', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample('foo', 5));
    dataset.addExample(getFakeExample('bar', 6));
    dataset.addExample(getFakeExample('foo', 7));

    const {xs, ys} =
        dataset.getData(null, {numFrames: 5, hopFrames: 5, shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(xs.shape).toEqual([3, 5, FAKE_FRAME_SIZE, 1]);
    expectTensorsClose(ys, tf.tensor2d([[1, 0], [0, 1], [0, 1]]));
  });

  it('Ragged example lengths and one window per example, with label', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample('foo', 5));
    dataset.addExample(getFakeExample('bar', 6));
    dataset.addExample(getFakeExample('foo', 7));

    const {xs, ys} = dataset.getData('foo', {numFrames: 5, hopFrames: 5}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(xs.shape).toEqual([2, 5, FAKE_FRAME_SIZE, 1]);
    expect(ys).toBeUndefined();
  });

  it('Ragged example lengths and multiple windows per example', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const {xs, ys} =
        dataset.getData(
            null,
            {numFrames: 3, hopFrames: 1, shuffle: false, normalize: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    const windows = tf.unstack(xs);

    expect(windows.length).toEqual(6);
    expectTensorsClose(windows[0], tf.tensor3d([1, 1, 2, 2, 3, 3], [3, 2, 1]));
    expectTensorsClose(windows[1], tf.tensor3d([2, 2, 3, 3, 2, 2], [3, 2, 1]));
    expectTensorsClose(windows[2], tf.tensor3d([3, 3, 2, 2, 1, 1], [3, 2, 1]));
    expectTensorsClose(
        windows[3], tf.tensor3d([10, 10, 20, 20, 30, 30], [3, 2, 1]));
    expectTensorsClose(
        windows[4], tf.tensor3d([20, 20, 30, 30, 20, 20], [3, 2, 1]));
    expectTensorsClose(
        windows[5], tf.tensor3d([30, 30, 20, 20, 10, 10], [3, 2, 1]));
    expectTensorsClose(
        ys, tf.tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]));
  });

  it('getData with mixing-noise augmentation: get tensors', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        BACKGROUND_NOISE_TAG, 6, 2,
        [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const {xs, ys} = dataset.getData(null, {
      numFrames: 3,
      hopFrames: 1,
      shuffle: false,
      normalize: false,
      augmentByMixingNoiseRatio: 0.5
    }) as {xs: tf.Tensor, ys: tf.Tensor};

    // 3 of the newly-generated ones at the end are from the augmentation.
    expect(xs.shape).toEqual([10, 3, 2, 1]);
    expect(ys.shape).toEqual([10, 2]);
    const indices = tf.argMax(ys, -1).dataSync();
    const backgroundNoiseIndex = indices[0];
    for (let i = 0; i < 3; ++i) {
      expect(indices[indices.length - 1 - i] === backgroundNoiseIndex)
          .toEqual(false);
    }
  });

  // TODO(cais): Test that augmentByNoise without BACKGROUND_NOISE tag leads to
  // Error.

  it('getSpectrogramsAsTensors: normalize=true', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    // `normalize` is `true` by default.
    const {xs, ys} =
        dataset.getData(null, {numFrames: 3, hopFrames: 1, shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    const windows = tf.unstack(xs);

    expect(windows.length).toEqual(6);
    for (let i = 0; i < 6; ++i) {
      const {mean, variance} = tf.moments(windows[0]);
      expectTensorsClose(mean, tf.scalar(0));
      expectTensorsClose(variance, tf.scalar(1));
    }
    expectTensorsClose(
        windows[0], normalize(tf.tensor3d([1, 1, 2, 2, 3, 3], [3, 2, 1])));
    expectTensorsClose(
        windows[1], normalize(tf.tensor3d([2, 2, 3, 3, 2, 2], [3, 2, 1])));
    expectTensorsClose(
        windows[2], normalize(tf.tensor3d([3, 3, 2, 2, 1, 1], [3, 2, 1])));
    expectTensorsClose(
        windows[3],
        normalize(tf.tensor3d([10, 10, 20, 20, 30, 30], [3, 2, 1])));
    expectTensorsClose(
        windows[4],
        normalize(tf.tensor3d([20, 20, 30, 30, 20, 20], [3, 2, 1])));
    expectTensorsClose(
        windows[5],
        normalize(tf.tensor3d([30, 30, 20, 20, 10, 10], [3, 2, 1])));
    expectTensorsClose(
        ys, tf.tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]));
  });

  it('getSpectrogramsAsTensors: shuffle=true leads to random order', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const argMaxTensors: tf.Tensor[] = [];
    for (let i = 0; i < 5; ++i) {
      // `shuffle` is `true` by default.
      const {ys} = dataset.getData(null, {numFrames: 3, hopFrames: 1}) as
          {xs: tf.Tensor, ys: tf.Tensor};
      // Use `argMax()` to convert the one-hot-encoded `ys` to indices.
      argMaxTensors.push(tf.argMax(ys, -1));
    }
    // `argMaxTensors` are the indices for the targets.
    // We stack them into a 2D tensor and then calculate the variance along
    // the examples dimension, in order to detect variance in the indices
    // between the iterations above.
    const argMaxMerged = tf.stack(argMaxTensors);
    // Assert that the orders are not all the same. This is asserting that
    // the indices are not all the same among the 5 iterations above.
    const {variance} = tf.moments(argMaxMerged, 0);
    expect(tf.max(variance).dataSync()[0]).toBeGreaterThan(0);
  });

  it('getSpectrogramsAsTensors: shuffle=false leads to constant order', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const argMaxTensors: tf.Tensor[] = [];
    for (let i = 0; i < 5; ++i) {
      const {ys} =
          dataset.getData(null, {numFrames: 3, hopFrames: 1, shuffle: false}) as
          {xs: tf.Tensor, ys: tf.Tensor};
      argMaxTensors.push(tf.argMax(ys, -1));
    }
    // `argMaxTensors` are the indices for the targets.
    // We stack them into a 2D tensor and then calculate the variance along
    // the examples dimension, in order to detect variance in the indices
    // between the iterations above.
    const argMaxMerged = tf.stack(argMaxTensors);
    // Assert that the orders are not all the same. This is asserting that
    // the indices are all the same among the 5 iterations above.
    const {variance} = tf.moments(argMaxMerged, 0);
    expect(tf.max(variance).dataSync()[0]).toEqual(0);
  });

  it('Uniform example lengths and multiple windows per example', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 6, 2, [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));

    const {xs, ys} =
        dataset.getData(
            null,
            {numFrames: 5, hopFrames: 1, shuffle: false, normalize: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    const windows = tf.unstack(xs);
    expect(windows.length).toEqual(4);
    expectTensorsClose(
        windows[0], tf.tensor3d([0, 0, 1, 1, 2, 2, 3, 3, 2, 2], [5, 2, 1]));
    expectTensorsClose(
        windows[1], tf.tensor3d([1, 1, 2, 2, 3, 3, 2, 2, 1, 1], [5, 2, 1]));
    expectTensorsClose(
        windows[2],
        tf.tensor3d([10, 10, 20, 20, 30, 30, 20, 20, 10, 10], [5, 2, 1]));
    expectTensorsClose(
        windows[3],
        tf.tensor3d([20, 20, 30, 30, 20, 20, 10, 10, 0, 0], [5, 2, 1]));
    expectTensorsClose(ys, tf.tensor2d([[1, 0], [1, 0], [0, 1], [0, 1]]));
  });

  it('Ragged examples containing background noise', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        BACKGROUND_NOISE_TAG, 7, 2,
        [0, 0, 10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 6, 2, [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));
    const {xs, ys} =
        dataset.getData(
            null,
            {numFrames: 3, hopFrames: 2, shuffle: false, normalize: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    const windows = tf.unstack(xs);
    expect(windows.length).toEqual(4);
    expectTensorsClose(
        windows[0], tf.tensor3d([0, 0, 10, 10, 20, 20], [3, 2, 1]));
    expectTensorsClose(
        windows[1], tf.tensor3d([20, 20, 30, 30, 20, 20], [3, 2, 1]));
    expectTensorsClose(
        windows[2], tf.tensor3d([20, 20, 10, 10, 0, 0], [3, 2, 1]));
    expectTensorsClose(windows[3], tf.tensor3d([2, 2, 3, 3, 2, 2], [3, 2, 1]));
    expectTensorsClose(ys, tf.tensor2d([[1, 0], [1, 0], [1, 0], [0, 1]]));
  });

  it('numFrames exceeding minmum example length leads to Error', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));
    expect(() => dataset.getData(null, {numFrames: 6, hopFrames: 2}))
        .toThrowError(/.*6.*exceeds the minimum numFrames .*5.*/);
  });

  it('Ragged examples with no numFrames leads to Error', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));
    expect(() => dataset.getData(null)).toThrowError(/numFrames is required/);
  });

  it('Ragged examples with no hopFrames leads to Error', () => {
    const dataset = new Dataset();
    dataset.addExample(getFakeExample(
        'foo', 6, 2, [10, 10, 20, 20, 30, 30, 20, 20, 10, 10, 0, 0]));
    dataset.addExample(
        getFakeExample('bar', 5, 2, [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]));
    expect(() => dataset.getData(null, {
      numFrames: 4
    })).toThrowError(/hopFrames is required/);
  });
});

describe('Dataset serialization', () => {
  function getRandomExample(
      label: string, numFrames: number, frameSize: number,
      rawAudioNumSamples?: number, rawAudioSampleRateHz?: number): Example {
    const spectrogramData = [];
    for (let i = 0; i < numFrames * frameSize; ++i) {
      spectrogramData.push(Math.random());
    }
    const output: Example = {
      label,
      spectrogram: {data: new Float32Array(spectrogramData), frameSize}
    };
    if (rawAudioNumSamples != null) {
      const rawAudioData: number[] = [];
      for (let i = 0; i < rawAudioNumSamples; ++i) {
        rawAudioData.push(Math.random());
      }
      const rawAudio: RawAudioData = {
        data: new Float32Array(rawAudioData),
        sampleRateHz: rawAudioSampleRateHz
      };
      output.rawAudio = rawAudio;
    }
    return output;
  }

  it('serializeExample-deserializeExample round trip, no raw audio', () => {
    const label = 'foo';
    const numFrames = 10;
    const frameSize = 16;
    const ex = getRandomExample(label, numFrames, frameSize);
    const artifacts = serializeExample(ex);
    expect(artifacts.spec.label).toEqual(label);
    expect(artifacts.spec.spectrogramNumFrames).toEqual(numFrames);
    expect(artifacts.spec.spectrogramFrameSize).toEqual(frameSize);
    expect(artifacts.spec.rawAudioNumSamples).toBeUndefined();
    expect(artifacts.spec.rawAudioSampleRateHz).toBeUndefined();
    expect(artifacts.data.byteLength).toEqual(4 * numFrames * frameSize);

    const exPrime = deserializeExample(artifacts);
    expect(exPrime.label).toEqual(ex.label);
    expect(exPrime.spectrogram.frameSize).toEqual(ex.spectrogram.frameSize);
    test_util.expectArraysEqual(exPrime.spectrogram.data, ex.spectrogram.data);
  });

  it('serializeExample-deserializeExample round trip, with raw audio', () => {
    const label = 'foo';
    const numFrames = 10;
    const frameSize = 16;
    const rawAudioNumSamples = 200;
    const rawAudioSampleRateHz = 48000;
    const ex = getRandomExample(
        label, numFrames, frameSize, rawAudioNumSamples, rawAudioSampleRateHz);
    const artifacts = serializeExample(ex);
    expect(artifacts.spec.label).toEqual(label);
    expect(artifacts.spec.spectrogramNumFrames).toEqual(numFrames);
    expect(artifacts.spec.spectrogramFrameSize).toEqual(frameSize);
    expect(artifacts.spec.rawAudioNumSamples).toEqual(rawAudioNumSamples);
    expect(artifacts.spec.rawAudioSampleRateHz).toEqual(rawAudioSampleRateHz);
    expect(artifacts.data.byteLength)
        .toEqual(4 * (numFrames * frameSize + rawAudioNumSamples));

    const exPrime = deserializeExample(artifacts);
    expect(exPrime.label).toEqual(ex.label);
    expect(exPrime.spectrogram.frameSize).toEqual(ex.spectrogram.frameSize);
    expect(exPrime.rawAudio.sampleRateHz).toEqual(ex.rawAudio.sampleRateHz);
    test_util.expectArraysEqual(exPrime.spectrogram.data, ex.spectrogram.data);
    test_util.expectArraysEqual(exPrime.rawAudio.data, ex.rawAudio.data);
  });

  it('Dataset.serialize()', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 12, 16);
    const ex3 = getRandomExample('qux', 14, 16);
    const ex4 = getRandomExample('foo', 13, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);
    const buffer = dataset.serialize();
    const {manifest, data} = arrayBuffer2SerializedExamples(buffer);
    expect(manifest).toEqual([
      {label: 'bar', spectrogramNumFrames: 12, spectrogramFrameSize: 16},
      {label: 'foo', spectrogramNumFrames: 10, spectrogramFrameSize: 16},
      {label: 'foo', spectrogramNumFrames: 13, spectrogramFrameSize: 16},
      {label: 'qux', spectrogramNumFrames: 14, spectrogramFrameSize: 16}
    ]);
    expect(data.byteLength).toEqual(4 * (10 + 12 + 14 + 13) * 16);
  });

  it('Dataset.serialize(): limited singleton word labels', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 12, 16);
    const ex3 = getRandomExample('qux', 14, 16);
    const ex4 = getRandomExample('foo', 13, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);

    let buffer = dataset.serialize('foo');
    let loaded = arrayBuffer2SerializedExamples(buffer);
    expect(loaded.manifest).toEqual([
      {label: 'foo', spectrogramNumFrames: 10, spectrogramFrameSize: 16},
      {label: 'foo', spectrogramNumFrames: 13, spectrogramFrameSize: 16}
    ]);
    expect(loaded.data.byteLength).toEqual(4 * (10 + 13) * 16);

    buffer = dataset.serialize('bar');
    loaded = arrayBuffer2SerializedExamples(buffer);
    expect(loaded.manifest).toEqual([
      {label: 'bar', spectrogramNumFrames: 12, spectrogramFrameSize: 16}
    ]);
    expect(loaded.data.byteLength).toEqual(4 * 12 * 16);
  });

  it('Dataset.serialize(): limited array word label', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 12, 16);
    const ex3 = getRandomExample('qux', 14, 16);
    const ex4 = getRandomExample('foo', 13, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);

    const buffer = dataset.serialize(['foo', 'qux']);
    const loaded = arrayBuffer2SerializedExamples(buffer);
    expect(loaded.manifest).toEqual([
      {label: 'foo', spectrogramNumFrames: 10, spectrogramFrameSize: 16},
      {label: 'foo', spectrogramNumFrames: 13, spectrogramFrameSize: 16},
      {label: 'qux', spectrogramNumFrames: 14, spectrogramFrameSize: 16}
    ]);
    expect(loaded.data.byteLength).toEqual(4 * (10 + 13 + 14) * 16);
  });

  it('Dataset.serialize(): nonexistent singleton word label errors', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 12, 16);
    const ex3 = getRandomExample('qux', 14, 16);
    const ex4 = getRandomExample('foo', 13, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);

    expect(() => dataset.serialize('gralk'))
        .toThrowError(/\"gralk\" does not exist/);
    expect(() => dataset.serialize(['gralk']))
        .toThrowError(/\"gralk\" does not exist/);
    expect(() => dataset.serialize([
      'foo', 'gralk'
    ])).toThrowError(/\"gralk\" does not exist/);
  });

  it('Dataset serialize-deserialize round trip', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 10, 16);
    const ex3 = getRandomExample('qux', 10, 16);
    const ex4 = getRandomExample('foo', 10, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);

    const artifacts = dataset.serialize();
    const datasetPrime = new Dataset(artifacts);

    expect(datasetPrime.empty()).toEqual(false);
    expect(datasetPrime.size()).toEqual(4);
    expect(datasetPrime.getVocabulary()).toEqual(['bar', 'foo', 'qux']);
    expect(dataset.getExampleCounts()).toEqual({'bar': 1, 'foo': 2, 'qux': 1});

    expect(dataset.getExamples('bar').length).toEqual(1);
    expect(dataset.getExamples('foo').length).toEqual(2);
    expect(dataset.getExamples('qux').length).toEqual(1);

    const ex1Prime = datasetPrime.getExamples('foo')[0].example;
    expect(ex1Prime.label).toEqual('foo');
    expect(ex1Prime.spectrogram.frameSize).toEqual(16);
    test_util.expectArraysEqual(
        ex1Prime.spectrogram.data, ex1.spectrogram.data);

    const ex2Prime = datasetPrime.getExamples('bar')[0].example;
    expect(ex2Prime.label).toEqual('bar');
    expect(ex2Prime.spectrogram.frameSize).toEqual(16);
    test_util.expectArraysEqual(
        ex2Prime.spectrogram.data, ex2.spectrogram.data);

    const ex3Prime = datasetPrime.getExamples('qux')[0].example;
    expect(ex3Prime.label).toEqual('qux');
    expect(ex3Prime.spectrogram.frameSize).toEqual(16);
    test_util.expectArraysEqual(
        ex3Prime.spectrogram.data, ex3.spectrogram.data);

    const ex4Prime = datasetPrime.getExamples('foo')[1].example;
    expect(ex4Prime.label).toEqual('foo');
    expect(ex4Prime.spectrogram.frameSize).toEqual(16);
    test_util.expectArraysEqual(
        ex4Prime.spectrogram.data, ex4.spectrogram.data);

    const {xs, ys} = datasetPrime.getData(null, {shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(xs.shape).toEqual([4, 10, 16, 1]);
    expectTensorsClose(
        ys, tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]));
  });

  it('Calling serialize() on empty dataset fails', () => {
    const dataset = new Dataset();
    expect(() => dataset.serialize())
        .toThrowError(/Cannot serialize empty Dataset/);
  });

  it('Deserialized dataset supports removeExample', () => {
    const dataset = new Dataset();
    const ex1 = getRandomExample('foo', 10, 16);
    const ex2 = getRandomExample('bar', 10, 16);
    const ex3 = getRandomExample('qux', 10, 16);
    const ex4 = getRandomExample('foo', 10, 16);
    dataset.addExample(ex1);
    dataset.addExample(ex2);
    dataset.addExample(ex3);
    dataset.addExample(ex4);

    const serialized = dataset.serialize();
    const datasetPrime = new Dataset(serialized);

    const examples = datasetPrime.getExamples('foo');
    datasetPrime.removeExample(examples[0].uid);

    const {xs, ys} = datasetPrime.getData(null, {shuffle: false}) as
        {xs: tf.Tensor, ys: tf.Tensor};
    expect(xs.shape).toEqual([3, 10, 16, 1]);
    expectTensorsClose(ys, tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
  });

  it('Attempt to load invalid ArrayBuffer errors out', () => {
    const invalidBuffer = string2ArrayBuffer('INVALID_[{}]0000000');
    expect(() => new Dataset(invalidBuffer))
        .toThrowError('Deserialization error: Invalid descriptor');
  });

  it('DATASET_SERIALIZATION_DESCRIPTOR has right length', () => {
    expect(DATASET_SERIALIZATION_DESCRIPTOR.length).toEqual(8);
    expect(string2ArrayBuffer(DATASET_SERIALIZATION_DESCRIPTOR).byteLength)
        .toEqual(8);
  });

  it('Version number satisfies requirements', () => {
    expect(typeof DATASET_SERIALIZATION_VERSION === 'number').toEqual(true);
    expect(Number.isInteger(DATASET_SERIALIZATION_VERSION)).toEqual(true);
    expect(DATASET_SERIALIZATION_VERSION).toBeGreaterThan(0);
  });
});

describe('getValidWindows', () => {
  it('Left and right sides open, odd windowLength', () => {
    const snippetLength = 100;
    const focusIndex = 50;
    const windowLength = 21;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[30, 51], [35, 56], [40, 61], [45, 66], [50, 71]]);
  });

  it('Left and right sides open, even windowLength', () => {
    const snippetLength = 100;
    const focusIndex = 50;
    const windowLength = 20;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[35, 55], [40, 60], [45, 65], [50, 70]]);
  });

  it('Left side truncation, right side open', () => {
    const snippetLength = 100;
    const focusIndex = 8;
    const windowLength = 20;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[0, 20], [5, 25]]);
  });

  it('Left side truncation extreme, right side open', () => {
    const snippetLength = 100;
    const focusIndex = 0;
    const windowLength = 21;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[0, 21]]);
  });

  it('Right side truncation, left side open', () => {
    const snippetLength = 100;
    const focusIndex = 95;
    const windowLength = 20;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[80, 100]]);
  });

  it('Right side truncation extreme, left side open', () => {
    const snippetLength = 100;
    const focusIndex = 99;
    const windowLength = 21;
    const windowHop = 5;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[79, 100]]);
  });

  it('Neither side has enough room for another hop 1', () => {
    const snippetLength = 100;
    const focusIndex = 50;
    const windowLength = 21;
    const windowHop = 35;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[40, 61]]);
  });

  it('Neither side has enough room for another hop 2', () => {
    const snippetLength = 100;
    const focusIndex = 50;
    const windowLength = 91;
    const windowHop = 35;
    const windows =
        getValidWindows(snippetLength, focusIndex, windowLength, windowHop);
    expect(windows).toEqual([[5, 96]]);
  });

  it('Exact match', () => {
    const snippetLength = 10;
    const windowLength = 10;
    const windowHop = 2;

    let focusIndex = 0;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 1;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 5;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 8;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 9;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
  });

  it('Almost exact match', () => {
    const snippetLength = 12;
    const windowLength = 10;
    const windowHop = 2;

    let focusIndex = 0;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 1;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10]]);
    focusIndex = 5;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10], [2, 12]]);
    focusIndex = 8;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10], [2, 12]]);
    focusIndex = 9;
    expect(getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toEqual([[0, 10], [2, 12]]);
  });

  it('Non-positive integer snippetLength values lead to errors', () => {
    const windowLength = 10;
    const focusIndex = 5;
    const windowHop = 2;
    let snippetLength = 0;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    snippetLength = -2;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    snippetLength = 10.5;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
  });

  it('Non-positive integer windowLength values lead to errors', () => {
    const snippetLength = 10;
    const focusIndex = 5;
    const windowHop = 2;
    let windowLength = 0;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    windowLength = -2;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    windowLength = 3.5;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
  });

  it('Negative or non-integer focusIndex values lead to errors', () => {
    const snippetLength = 10;
    const windowLength = 10;
    const windowHop = 2;
    let focusIndex = -5;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    focusIndex = 1.5;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
  });

  it('Out-of-bound focusIndex leads to error', () => {
    const snippetLength = 10;
    const windowLength = 10;
    const windowHop = 2;
    let focusIndex = 10;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
    focusIndex = 11;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
  });

  it('Out-of-bound windowLength leads to error', () => {
    const snippetLength = 10;
    const windowLength = 12;
    const windowHop = 2;
    const focusIndex = 5;
    expect(
        () =>
            getValidWindows(snippetLength, focusIndex, windowLength, windowHop))
        .toThrow();
  });
});

describe('spectrogram2IntensityCurve', () => {
  it('Correctness', () => {
    const x = tf.tensor2d([[1, 2], [3, 4], [5, 6]]);
    const spectrogram:
        SpectrogramData = {data: x.dataSync() as Float32Array, frameSize: 2};
    const intensityCurve = spectrogram2IntensityCurve(spectrogram);
    expectTensorsClose(intensityCurve, tf.tensor1d([1.5, 3.5, 5.5]));
  });
});

describe('getMaxIntensityFrameIndex', () => {
  it('Multiple frames', () => {
    const x = tf.tensor2d([[1, 2], [11, 12], [3, 4], [51, 52], [5, 6]]);
    const spectrogram:
        SpectrogramData = {data: x.dataSync() as Float32Array, frameSize: 2};
    const maxIntensityFrameIndex = getMaxIntensityFrameIndex(spectrogram);
    expectTensorsClose(maxIntensityFrameIndex, tf.scalar(3, 'int32'));
  });

  it('Only one frames', () => {
    const x = tf.tensor2d([[11, 12]]);
    const spectrogram:
        SpectrogramData = {data: x.dataSync() as Float32Array, frameSize: 2};
    const maxIntensityFrameIndex = getMaxIntensityFrameIndex(spectrogram);
    expectTensorsClose(maxIntensityFrameIndex, tf.scalar(0, 'int32'));
  });

  it('No focus frame: return multiple windows', () => {
    const snippetLength = 100;
    const windowLength = 40;
    const windowHop = 20;
    const windows =
        getValidWindows(snippetLength, null, windowLength, windowHop);
    expect(windows).toEqual([[0, 40], [20, 60], [40, 80], [60, 100]]);
  });

  it('No focus frame: return one window', () => {
    const snippetLength = 10;
    const windowLength = 10;
    const windowHop = 2;
    const windows =
        getValidWindows(snippetLength, null, windowLength, windowHop);
    expect(windows).toEqual([[0, 10]]);
  });
});
