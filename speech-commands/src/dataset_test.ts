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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {Dataset} from './dataset';
import {Example} from './types';

describe('Dataset', () => {
  const fakeNumFrames = 4;
  const fakeFrameSize = 16;

  function getRandomExample(label: string): Example {
    const spectrogramData = [];
    for (let i = 0; i < fakeNumFrames * fakeFrameSize; ++i) {
      spectrogramData.push(Math.random());
    }
    return {
      label,
      spectrogram:
          {data: new Float32Array(spectrogramData), frameSize: fakeFrameSize}
    };
  }

  function addThreeExamplesToDataset(
      dataset: Dataset, labelA = 'a', labelB = 'b'): string[] {
    const ex1 = getRandomExample(labelA);
    const uid1 = dataset.addExample(ex1);
    const ex2 = getRandomExample(labelA);
    const uid2 = dataset.addExample(ex2);
    const ex3 = getRandomExample(labelB);
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
    const ex1 = getRandomExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(uid1.length).toBeGreaterThan(0);
    uids.push(uid1);
    console.log(uid1);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(1);
    expect(dataset.getExampleCounts()).toEqual({'a': 1});

    const ex2 = getRandomExample('a');
    const uid2 = dataset.addExample(ex2);
    expect(uids.indexOf(uid2)).toEqual(-1);
    uids.push(uid2);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(2);
    expect(dataset.getExampleCounts()).toEqual({'a': 2});

    const ex3 = getRandomExample('b');
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
    expect(() => dataset.addExample(getRandomExample(null)))
        .toThrowError(/Expected label to be a non-empty string.*null/);
    expect(() => dataset.addExample(getRandomExample(undefined)))
        .toThrowError(/Expected label to be a non-empty string.*undefined/);
    expect(() => dataset.addExample(getRandomExample('')))
        .toThrowError(/Expected label to be a non-empty string/);
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

    const ex = getRandomExample('a');
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

    const ex = getRandomExample('a');
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

    const ex1 = getRandomExample('a');
    const uid1 = dataset.addExample(ex1);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(1);
    expect(dataset.getExampleCounts()).toEqual({'a': 1});

    const ex2 = getRandomExample('a');
    const uid2 = dataset.addExample(ex2);
    expect(dataset.empty()).toEqual(false);
    expect(dataset.size()).toEqual(2);
    expect(dataset.getExampleCounts()).toEqual({'a': 2});

    const ex3 = getRandomExample('b');
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

    const ex1 = getRandomExample('a');
    const uid1 = dataset.addExample(ex1);
    dataset.removeExample(uid1);
    expect(() => dataset.removeExample(uid1))
        .toThrowError(/Nonexistent example UID/);
  });

  it('getVocabulary', () => {
    const dataset = new Dataset();
    expect(dataset.getVocabulary()).toEqual([]);

    const ex1 = getRandomExample('a');
    const ex2 = getRandomExample('a');
    const ex3 = getRandomExample('b');

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

    const out1 = dataset.getSpectrogramsAsTensors('a');
    expect(out1.xs.shape).toEqual([2, fakeNumFrames, fakeFrameSize, 1]);
    expect(out1.ys).toBeUndefined();
    const out2 = dataset.getSpectrogramsAsTensors('b');
    expect(out2.xs.shape).toEqual([1, fakeNumFrames, fakeFrameSize, 1]);
    expect(out2.ys).toBeUndefined();
  });

  it('getSpectrogramsAsTensors after removeExample', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);

    dataset.removeExample(uid1);
    const out1 = dataset.getSpectrogramsAsTensors();
    expect(out1.xs.shape).toEqual([2, fakeNumFrames, fakeFrameSize, 1]);
    expectArraysClose(out1.ys, tf.tensor2d([[1, 0], [0, 1]]));

    const out2 = dataset.getSpectrogramsAsTensors('a');
    expect(out2.xs.shape).toEqual([1, fakeNumFrames, fakeFrameSize, 1]);

    dataset.removeExample(uid2);
    expect(() => dataset.getSpectrogramsAsTensors('a'))
        .toThrowError(/Label a is not in the vocabulary/);

    const out3 = dataset.getSpectrogramsAsTensors('b');
    expect(out3.xs.shape).toEqual([1, fakeNumFrames, fakeFrameSize, 1]);
  });

  it('getSpectrogramsAsTensors w/o label on one-word vocabulary fails', () => {
    const dataset = new Dataset();
    const [uid1, uid2] = addThreeExamplesToDataset(dataset);
    dataset.removeExample(uid1);
    dataset.removeExample(uid2);

    expect(() => dataset.getSpectrogramsAsTensors())
        .toThrowError(/requires .* at least two words/);
  });
  
  it('getSpectrogramsAsTensors without label', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    const out = dataset.getSpectrogramsAsTensors();
    expect(out.xs.shape).toEqual([3, fakeNumFrames, fakeFrameSize, 1]);
    expectArraysClose(out.ys, tf.tensor2d([[1, 0], [1, 0], [0, 1]]));
  });

  it('getSpectrogramsAsTensors on nonexistent label fails', () => {
    const dataset = new Dataset();
    addThreeExamplesToDataset(dataset);

    expect(() => dataset.getSpectrogramsAsTensors('label3'))
        .toThrowError(/Label label3 is not in the vocabulary/);
  });

  it('getSpectrogramsAsTensors on empty Dataset fails', () => {
    const dataset = new Dataset();
    expect(() => dataset.getSpectrogramsAsTensors())
        .toThrowError(/Cannot get spectrograms as tensors because.*empty/);
  });
});
