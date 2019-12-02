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

import {stringToChars} from '../util';

// [token, score, index]
type OutputNode = [string[], number, number];

class TrieNode {
  public parent: TrieNode;
  public end: boolean;
  public children: {[firstSymbol: string]: TrieNode};
  public word: OutputNode;

  constructor() {
    this.parent = null;
    this.children = {};
    this.end = false;
    this.word = [[], 0, 0];
  }
}

export class Trie {
  public root: TrieNode;

  constructor() {
    this.root = new TrieNode();
  }

  /**
   * Inserts a token into the trie.
   */
  insert(word: string, score: number, index: number) {
    let node = this.root;

    const symbols = stringToChars(word);

    for (let i = 0; i < symbols.length; i++) {
      if (!node.children[symbols[i]]) {
        node.children[symbols[i]] = new TrieNode();
        node.children[symbols[i]].parent = node;
        node.children[symbols[i]].word[0] = node.word[0].concat(symbols[i]);
      }

      node = node.children[symbols[i]];
      if (i === symbols.length - 1) {
        node.end = true;
        node.word[1] = score;
        node.word[2] = index;
      }
    }
  }

  /**
   * Returns an array of all tokens starting with ss.
   *
   * @param ss The prefix to match on.
   */
  commonPrefixSearch(ss: string[]): OutputNode[] {
    const output: OutputNode[] = [];
    let node = this.root.children[ss[0]];

    for (let i = 0; i < ss.length && node; i++){
      if (node.end){ output.push(node.word); }
      node = node.children[ss[i + 1]];
    }

    if (!output.length){ output.push([[ss[0]], 0, 0]); }

    return output;
  }
}
