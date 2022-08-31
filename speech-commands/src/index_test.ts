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

// tslint:disable-next-line:no-require-imports
const packageJSON = require('../package.json');
import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import * as speechCommands from './index';

describe('Public API', () => {
  it('version matches package.json', () => {
    expect(typeof speechCommands.version).toEqual('string');
    expect(speechCommands.version).toEqual(packageJSON.version);
  });
});

describe('Creating recognizer', () => {
  async function makeModelArtifacts(): Promise<tf.io.ModelArtifacts> {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2d({
      filters: 8,
      kernelSize: 3,
      activation: 'relu',
      inputShape: [86, 500, 1]
    }));
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({units: 3, activation: 'softmax'}));
    let modelArtifacts: tf.io.ModelArtifacts;
    await model.save(tf.io.withSaveHandler(artifacts => {
      modelArtifacts = artifacts;
      return null;
    }));
    return modelArtifacts;
  }

  function makeMetadata(): speechCommands.SpeechCommandRecognizerMetadata {
    return {
      wordLabels: [speechCommands.BACKGROUND_NOISE_TAG, 'foo', 'bar'],
      tfjsSpeechCommandsVersion: speechCommands.version
    };
  }

  it('Create recognizer from aritfacts and metadata objects', async () => {
    const modelArtifacts = await makeModelArtifacts();
    const metadata = makeMetadata();
    const recognizer =
        speechCommands.create('BROWSER_FFT', null, modelArtifacts, metadata);
    await recognizer.ensureModelLoaded();

    expect(recognizer.wordLabels()).toEqual([
      speechCommands.BACKGROUND_NOISE_TAG, 'foo', 'bar'
    ]);
    expect(recognizer.modelInputShape()).toEqual([null, 86, 500, 1]);
  });
});
