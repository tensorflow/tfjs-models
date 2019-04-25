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

import {stubbedTokenizerVocab} from './test_util';
import {Tokenizer} from './tokenizer';

describe('Universal Sentence Encoder tokenizer', () => {
  let tokenizer: Tokenizer;
  beforeAll(() => {
    tokenizer = new Tokenizer(stubbedTokenizerVocab as Array<[string, number]>);
  });

  it('basic usage', () => {
    expect(tokenizer.encode('Ilikeit.')).toEqual([11, 15, 16, 10]);
  });

  it('handles whitespace', () => {
    expect(tokenizer.encode('I like it.')).toEqual([11, 12, 13, 10]);
  });

  it('should normalize inputs', () => {
    expect(tokenizer.encode('ça')).toEqual(tokenizer.encode('c\u0327a'));
  });

  it('should handle unknown inputs', () => {
    expect(() => tokenizer.encode('😹')).not.toThrow();
  });

  it('should treat consecutive unknown inputs as a single word', () => {
    expect(tokenizer.encode('a😹😹')).toEqual([7, 0]);
  });
});
