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

// [token, score, index]
type OutputNode = [string[], number, number];

class TrieNode {
  public parent: TrieNode;
  public end: boolean;
  public children: {[firstSymbol: string]: TrieNode};
  public score: number;
  public index: number;

  private key: string;

  constructor(key: string) {
    this.key = key;
    this.parent = null;
    this.children = {};
    this.end = false;
  }

  getWord(): OutputNode {
    const output: string[] = [];
    let node: TrieNode = this;

    while (node !== null) {
      if (node.key !== null) {
        output.unshift(node.key);
      }
      node = node.parent;
    }

    return [output, this.score, this.index];
  }
}

export class Trie {
  private root: TrieNode;

  constructor() {
    this.root = new TrieNode(null);
  }

  findAllCommonPrefixes(ss: string[], node: TrieNode, arr: OutputNode[]) {
    if (node.end) {
      const word = node.getWord();
      if (ss.slice(0, word[0].length).join('') === word[0].join('')) {
        arr.unshift(word);
      }
    }

    for (const child in node.children) {
      this.findAllCommonPrefixes(ss, node.children[child], arr);
    }
  }

  insert(word: string, score: number, index: number) {
    let node = this.root;

    const symbols = spreadSymbols(word);

    for (let i = 0; i < symbols.length; i++) {
      if (!node.children[symbols[i]]) {
        node.children[symbols[i]] = new TrieNode(symbols[i]);
        node.children[symbols[i]].parent = node;
      }

      node = node.children[symbols[i]];

      if (i === symbols.length - 1) {
        node.end = true;
        node.score = score;
        node.index = index;
      }
    }
  }

  commonPrefixSearch(ss: string[]) {
    const node = this.root.children[ss[0]];
    const output: OutputNode[] = [];
    if (node) {
      this.findAllCommonPrefixes(ss, node, output);
    } else {
      output.push([[ss[0]], 0, 0]);  // unknown token
    }
    return output;
  }
}
