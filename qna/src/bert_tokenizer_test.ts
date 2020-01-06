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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {expectArrayBuffersEqual} from '@tensorflow/tfjs-core/dist/test_util';

import {loadTokenizer} from './bert_tokenizer';

describeWithFlags('bertTokenizer', NODE_ENVS, () => {
  beforeEach(() => {});

  it('should load', async () => {
    const tokenizer = await loadTokenizer();

    expect(tokenizer).toBeDefined();
  });

  it('should tokenize', async () => {
    const tokenizer = await loadTokenizer();
    const result = tokenizer.tokenize('a new test');

    expect(result).toEqual([1037, 2047, 3231]);
  });

  it('should tokenize punctuation', async () => {
    const tokenizer = await loadTokenizer();
    const result = tokenizer.tokenize('a new [test]');

    expect(result).toEqual([1037, 2047, 1031, 3231, 1033]);
  });

  it('should cleanup control characters', async () => {
    const tokenizer = await loadTokenizer();
    const result = tokenizer.tokenize('a new\b\v [test]');

    expect(result).toEqual([1037, 2047, 1031, 3231, 1033]);
  });
});
