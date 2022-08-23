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

  it('Trie creates a child for each unique prefix', () => {
    const childKeys = Object.keys(tokenizer.trie.root.children);
    expect(childKeys).toEqual(['▁', 'a', '.', 'I', 'l', 'i', 'k', 'e', 't']);
  });

  it('Trie commonPrefixSearch basic usage', () => {
    const commonPrefixes =
        tokenizer.trie.commonPrefixSearch(['l', 'i', 'k', 'e'])
            .map(d => d[0].join(''));

    expect(commonPrefixes).toEqual(['l', 'like']);
  });
});
