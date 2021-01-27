/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const SEPERATOR = '\u2581';
export const UNK_INDEX = 100;
export const CLS_INDEX = 101;
export const CLS_TOKEN = '[CLS]';
export const SEP_INDEX = 102;
export const SEP_TOKEN = '[SEP]';
export const NFKC_TOKEN = 'NFKC';
export const VOCAB_BASE =
    'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1/';
export const VOCAB_URL = VOCAB_BASE + 'processed_vocab.json?tfjs-format=file';
/**
 * Class for represent node for token parsing Trie data structure.
 */
class TrieNode {
  parent: TrieNode;
  children: {[key: string]: TrieNode} = {};
  end = false;
  score: number;
  index: number;
  constructor(private key: string) {}

  getWord(): [string[], number, number] {
    const output: string[] = [];
    let node: TrieNode = this;

    while (node != null) {
      if (node.key != null) {
        output.unshift(node.key);
      }
      node = node.parent;
    }

    return [output, this.score, this.index];
  }
}

class Trie {
  private root = new TrieNode(null);

  /**
   * Insert the bert vacabulary word into the trie.
   * @param word word to be inserted.
   * @param score word score.
   * @param index index of word in the bert vocabulary file.
   */
  insert(word: string, score: number, index: number) {
    let node = this.root;

    const symbols = [];
    for (const symbol of word) {
      symbols.push(symbol);
    }

    for (let i = 0; i < symbols.length; i++) {
      if (node.children[symbols[i]] == null) {
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
   * @param token string, input string to be searched.
   */
  find(token: string): TrieNode {
    let node = this.root;
    let iter = 0;

    while (iter < token.length && node != null) {
      node = node.children[token[iter]];
      iter++;
    }

    return node;
  }
}

function isWhitespace(ch: string): boolean {
  return /\s/.test(ch);
}

function isInvalid(ch: string): boolean {
  return (ch.charCodeAt(0) === 0 || ch.charCodeAt(0) === 0xfffd);
}

const punctuations = '[~`!@#$%^&*(){}[];:"\'<,.>?/\\|-_+=';

/** To judge whether it's a punctuation. */
function isPunctuation(ch: string): boolean {
  return punctuations.indexOf(ch) !== -1;
}

export interface Token {
  text: string;
  index: number;
}
/**
 * Tokenizer for Bert.
 */
export class BertTokenizer {
  private vocab: string[];
  private trie: Trie;

  /**
   * Load the vacabulary file and initialize the Trie for lookup.
   */
  async load() {
    this.vocab = await this.loadVocab();

    this.trie = new Trie();
    // Actual tokens start at 999.
    for (let vocabIndex = 999; vocabIndex < this.vocab.length; vocabIndex++) {
      const word = this.vocab[vocabIndex];
      this.trie.insert(word, 1, vocabIndex);
    }
  }

  private async loadVocab(): Promise<[]> {
    return tf.util.fetch(VOCAB_URL).then(d => d.json());
  }

  processInput(text: string): Token[] {
    const charOriginalIndex: number[] = [];
    const cleanedText = this.cleanText(text, charOriginalIndex);
    const origTokens = cleanedText.split(' ');

    let charCount = 0;
    const tokens = origTokens.map((token) => {
      token = token.toLowerCase();
      const tokens = this.runSplitOnPunc(token, charCount, charOriginalIndex);
      charCount += token.length + 1;
      return tokens;
    });

    let flattenTokens: Token[] = [];
    for (let index = 0; index < tokens.length; index++) {
      flattenTokens = flattenTokens.concat(tokens[index]);
    }
    return flattenTokens;
  }

  /* Performs invalid character removal and whitespace cleanup on text. */
  private cleanText(text: string, charOriginalIndex: number[]): string {
    const stringBuilder: string[] = [];
    let originalCharIndex = 0, newCharIndex = 0;
    for (const ch of text) {
      // Skip the characters that cannot be used.
      if (isInvalid(ch)) {
        originalCharIndex += ch.length;
        continue;
      }
      if (isWhitespace(ch)) {
        if (stringBuilder.length > 0 &&
            stringBuilder[stringBuilder.length - 1] !== ' ') {
          stringBuilder.push(' ');
          charOriginalIndex[newCharIndex] = originalCharIndex;
          originalCharIndex += ch.length;
        } else {
          originalCharIndex += ch.length;
          continue;
        }
      } else {
        stringBuilder.push(ch);
        charOriginalIndex[newCharIndex] = originalCharIndex;
        originalCharIndex += ch.length;
      }
      newCharIndex++;
    }
    return stringBuilder.join('');
  }

  /* Splits punctuation on a piece of text. */
  private runSplitOnPunc(
      text: string, count: number,
      charOriginalIndex: number[]): Token[] {
    const tokens: Token[] = [];
    let startNewWord = true;
    for (const ch of text) {
      if (isPunctuation(ch)) {
        tokens.push({text: ch, index: charOriginalIndex[count]});
        count += ch.length;
        startNewWord = true;
      } else {
        if (startNewWord) {
          tokens.push({text: '', index: charOriginalIndex[count]});
          startNewWord = false;
        }
        tokens[tokens.length - 1].text += ch;
        count += ch.length;
      }
    }
    return tokens;
  }

  /**
   * Generate tokens for the given vocalbuary.
   * @param text text to be tokenized.
   */
  tokenize(text: string): number[] {
    // Source:
    // https://github.com/google-research/bert/blob/88a817c37f788702a363ff935fd173b6dc6ac0d6/tokenization.py#L311

    let outputTokens: number[] = [];

    const words = this.processInput(text);
    words.forEach(word => {
      if (word.text !== CLS_TOKEN && word.text !== SEP_TOKEN) {
        word.text = `${SEPERATOR}${word.text.normalize(NFKC_TOKEN)}`;
      }
    });

    for (let i = 0; i < words.length; i++) {
      const chars = [];
      for (const symbol of words[i].text) {
        chars.push(symbol);
      }

      let isUnknown = false;
      let start = 0;
      const subTokens: number[] = [];

      const charsLength = chars.length;

      while (start < charsLength) {
        let end = charsLength;
        let currIndex;

        while (start < end) {
          const substr = chars.slice(start, end).join('');

          const match = this.trie.find(substr);
          if (match != null && match.end != null) {
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
        outputTokens.push(UNK_INDEX);
      } else {
        outputTokens = outputTokens.concat(subTokens);
      }
    }

    return outputTokens;
  }
}

export async function loadTokenizer(): Promise<BertTokenizer> {
  const tokenizer = new BertTokenizer();
  await tokenizer.load();
  return tokenizer;
}
