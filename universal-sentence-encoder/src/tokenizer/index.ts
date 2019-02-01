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

import {spreadSymbols} from '../util';

import Trie from './trie';

function processInput(str: string): string {
  const normalized = str.normalize('NFKC');
  return '▁' + normalized.replace(/ /g, '▁');
}

type Vocabulary = Array<[string, number]>;

type Score = {
  key: string[],
  score: number,
  index: number
};

class Tokenizer {
  vocabulary: Vocabulary;
  trie: Trie;

  constructor(vocabulary: Vocabulary) {
    this.vocabulary = vocabulary;
    this.trie = new Trie();

    // The first five tokens are reserved for unk, control symbols, and
    // user-defined symbols.
    const reservedSymbolsCount = 6;

    for (let i = reservedSymbolsCount; i < this.vocabulary.length; i++) {
      this.trie.insert(this.vocabulary[i][0], this.vocabulary[i][1], i);
    }
  }

  encode(input: string) {
    const nodes: {[index: number]: Score[]}[] = [];
    const words: number[] = [];
    const best: number[] = [];

    input = processInput(input);

    const symbols = spreadSymbols(input);

    for (let i = 0; i <= symbols.length; i++) {
      nodes.push({});
      words.push(0);
      best.push(0);
    }

    for (let i = 0; i < symbols.length; i++) {
      const matches = this.trie.commonPrefixSearch(symbols.slice(i));

      for (let j = 0; j < matches.length; j++) {
        const piece = matches[j];
        const obj = {key: piece[0], score: piece[1], index: piece[2]};

        const endPos = piece[0].length;
        if (nodes[i + endPos][i] == null) {
          nodes[i + endPos][i] = [];
        }

        nodes[i + endPos][i].push(obj);
      }
    }


    for (let endPos = 0; endPos <= symbols.length; endPos++) {
      for (let startPos in nodes[endPos]) {
        const arr = nodes[endPos][startPos];

        for (let j = 0; j < arr.length; j++) {
          const word = arr[j];
          const score = word.score + best[endPos - word.key.length];

          if (best[endPos] === 0 || score >= best[endPos]) {
            best[endPos] = score;
            words[endPos] = arr[j].index;
          }
        }
      }
    }

    const results: number[] = [];

    let iter = words.length - 1;
    while (iter > 0) {
      results.push(words[iter]);
      iter -= this.vocabulary[words[iter]][0].length;
    }

    // Merge contiguous unks.
    const merged = [];
    let isPreviousUnk = false;
    for (let i = 0; i < results.length; i++) {
      const id = results[i];
      if (!(isPreviousUnk && id === 0)) {
        merged.push(id);
      }

      isPreviousUnk = id === 0;
    }

    return merged.reverse();
  }
}

export default Tokenizer;