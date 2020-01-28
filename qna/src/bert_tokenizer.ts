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

/**
 * Class for represent node for token parsing Trie data structure.
 */
class TrieNode {
  parent: TrieNode = null;
  children: {[key: string]: TrieNode} = {};
  end = false;
  score: number;
  index: number;
  constructor(private key: string) {}

  getWord() {
    const output = [];
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

class Trie {
  private root: TrieNode = new TrieNode(null);
  constructor() {}

  /**
   *
   * @param word string, word to be inserted.
   * @param score number: word score.
   * @param index number: index of word in the bert vocabulary file.
   */
  insert(word: string, score: number, index: number) {
    let node: TrieNode = this.root;

    const symbols = [];
    for (const symbol of word) {
      symbols.push(symbol);
    }

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

  /**
   * Find the Trie node for the given token, it will return the first node that
   * matches the subtoken from the beginning of the token.
   * @param token string, input string to be searched
   */
  find(token: string) {
    let node = this.root;
    let iter = 0;

    while (iter < token.length && node != null) {
      node = node.children[token[iter]];
      iter++;
    }

    return node;
  }
}

function isWhitespace(ch) {
  return /\s/.test(ch);
}

function isInvalid(ch) {
  return (ch === 0 || ch === 0xfffd);
}

const punctuationRegEx = /[~`\!@#\$%\^&\*\(\)\{\}\[\];:\"'<,\.>\?\/\\\|\-_\+=]/;

/** To judge whether it's a punctuation. */
function isPunctuation(ch) {
  return punctuationRegEx.test(ch);
}
/**
 * Tokenizer for Bert.
 */
export class BertTokenizer {
  separator = '\u2581';
  UNK_INDEX = 100;
  CLS_INDEX = 101;
  SEP_INDEX = 102;
  private vocab: string[];
  private trie: Trie;
  constructor() {}

  /**
   * Load the vacabulary file and initialize the Trie for lookup.
   */
  async load() {
    this.vocab = await this.loadVocab();

    this.trie = new Trie();
    // Actual tokens start at 999.
    for (let i = 999; i < this.vocab.length; i++) {
      const word = this.vocab[i];
      this.trie.insert(word, 1, i);
    }
  }

  private async loadVocab() {
    return tf.util
        .fetch(
            'https://storage.googleapis.com/learnjs-data/bert_vocab/processed_vocab.json')
        .then(d => d.json());
  }

  processInput(text: string) {
    const cleanedText = this.cleanText(text);

    const origTokens = cleanedText.split(' ');

    const tokens = origTokens.map((token) => {
      token = token.toLowerCase();
      return this.runSplitOnPunc(token);
    });
    return [].concat.apply([], tokens);
  }

  /* Performs invalid character removal and whitespace cleanup on text. */
  private cleanText(text: string) {
    if (text == null) {
      throw new Error('The input String is null.');
    }

    const stringBuilder = [];
    for (const ch of text) {
      // Skip the characters that cannot be used.
      if (isInvalid(ch)) {
        continue;
      }
      if (isWhitespace(ch)) {
        stringBuilder.push(' ');
      } else {
        stringBuilder.push(ch);
      }
    }
    return stringBuilder.join('');
  }

  /* Splits punctuation on a piece of text. */
  private runSplitOnPunc(text: string) {
    if (text == null) {
      throw new Error('The input String is null.');
    }

    const tokens = [];
    let startNewWord = true;
    for (const ch of text) {
      if (isPunctuation(ch)) {
        tokens.push(ch);
        startNewWord = true;
      } else {
        if (startNewWord) {
          tokens.push('');
          startNewWord = false;
        }
        tokens[tokens.length - 1] += ch;
      }
    }
    return tokens;
  }

  /**
   * Generate tokens for the given vocalbuary.
   * @param text string
   */
  tokenize(text) {
    // Source:
    // https://github.com/google-research/bert/blob/88a817c37f788702a363ff935fd173b6dc6ac0d6/tokenization.py#L311

    let outputTokens = [];

    const words = this.processInput(text).map(word => {
      if (word !== '[CLS]' && word !== '[SEP]') {
        return `${this.separator}${word.normalize('NFKC')}`;
      }
      return word;
    });
    for (let i = 0; i < words.length; i++) {
      const chars = [];
      for (const symbol of words[i]) {
        chars.push(symbol);
      }

      let isUnknown = false;
      let start = 0;
      const subTokens = [];

      const charsLength = chars.length;

      while (start < charsLength) {
        let end = charsLength;
        let currIndex;

        while (start < end) {
          const substr = chars.slice(start, end).join('');

          const match = this.trie.find(substr);
          if (match != null && match.end) {
            currIndex = match.getWord()[2];
            break;
          }

          end = end - 1;
        }

        if (currIndex == null) {
          isUnknown = true;
          break;
        }

        subTokens.push(currIndex);
        start = end;
      }

      if (isUnknown) {
        outputTokens.push(this.UNK_INDEX);
      } else {
        outputTokens = outputTokens.concat(subTokens);
      }
    }

    return outputTokens;
  }
}

export const loadTokenizer = async(): Promise<BertTokenizer> => {
  const tokenizer = new BertTokenizer();
  await tokenizer.load();
  return tokenizer;
};
