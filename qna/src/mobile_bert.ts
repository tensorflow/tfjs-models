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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {BertTokenizer, loadTokenizer} from './bert_tokenizer';

const BASE_DIR = 'https://storage.googleapis.com/tfjs-testing/mobile-bert/';
const MODEL_URL = BASE_DIR + 'model.json';
const size = 384;
const MAX_ANS_LEN = 32;
const MAX_QUERY_LEN = 64;
const MAX_SEQ_LEN = 384;
const PREDICT_ANS_NUM = 5;
const OUTPUT_OFFSET = 1;

export interface MobileBert {
  findAnswers(question: string, context: string): Array<[string, number]>;
}

interface Feature {
  inputIds: number[];
  inputMask: number[];
  segmentIds: number[];
  origTokens: string[];
  tokenToOrigMap: {[key: number]: number};
}

class MobileBertImpl implements MobileBert {
  private model: tfconv.GraphModel;
  private tokenizer: BertTokenizer;

  private process(
      query: string, context: string, maxQueryLen: number, maxSeqLen: number,
      docStride = 128): Feature[] {
    let queryTokens = this.tokenizer.tokenize(query);
    if (queryTokens.length > maxQueryLen) {
      queryTokens = queryTokens.slice(0, maxQueryLen);
    }
    const origTokens = context.trim().split(/\s+/).slice(0);
    const tokenToOrigIndex = [];
    const allDocTokens = [];
    for (let i = 0; i < origTokens.length; i++) {
      const token = origTokens[i];
      const subTokens = this.tokenizer.tokenize(token);
      for (let j = 0; j < subTokens.length; j++) {
        const subToken = subTokens[j];
        tokenToOrigIndex.push(i);
        allDocTokens.push(subToken);
      }
    }
    // The -3 accounts for [CLS], [SEP] and [SEP]
    const maxContextLen = maxSeqLen - queryTokens.length - 3;

    // We can have documents that are longer than the maximum sequence
    // length. To deal with this we do a sliding window approach, where we
    // take chunks of the up to our max length with a stride of
    // `doc_stride`.
    const docSpans = [];
    let startOffset = 0;
    while (startOffset < allDocTokens.length) {
      let length = allDocTokens.length - startOffset;
      if (length > maxContextLen) {
        length = maxContextLen;
      }
      docSpans.push({start: startOffset, length});
      if (startOffset + length === allDocTokens.length) {
        break;
      }
      startOffset += Math.min(length, docStride);
    }

    const features = docSpans.map(docSpan => {
      const tokens = [];
      const segmentIds = [];
      const tokenToOrigMap = {};
      tokens.push(this.tokenizer.CLS_INDEX);
      segmentIds.push(0);
      for (let i = 0; i < queryTokens.length; i++) {
        const queryToken = queryTokens[i];
        tokens.push(queryToken);
        segmentIds.push(0);
      }
      tokens.push(this.tokenizer.SEP_INDEX);
      segmentIds.push(0);
      for (let i = 0; i < docSpan['length']; i++) {
        const splitTokenIndex = i + docSpan['start'];
        const docToken = allDocTokens[splitTokenIndex];
        tokens.push(docToken);
        segmentIds.push(1);
        tokenToOrigMap[tokens.length] = tokenToOrigIndex[splitTokenIndex];
      }
      tokens.push(this.tokenizer.SEP_INDEX);
      segmentIds.push(1);
      const inputIds = tokens;
      const inputMask = inputIds.map(id => 1);
      while ((inputIds.length < maxSeqLen)) {
        inputIds.push(0);
        inputMask.push(0);
        segmentIds.push(0);
      }
      return {inputIds, inputMask, segmentIds, origTokens, tokenToOrigMap};
    });
    return features;
  }

  async load() {
    this.model = await tfconv.loadGraphModel(MODEL_URL);
    // warm up the backend
    const inputIds = tf.ones([1, size], 'int32');
    const segmentIds = tf.ones([1, size], 'int32');
    const inputMask = tf.ones([1, size], 'int32');

    this.model.predict({
      input_ids: inputIds,
      segment_ids: segmentIds,
      input_mask: inputMask,
      global_step: tf.scalar(1, 'int32')
    });

    this.tokenizer = await loadTokenizer();
  }

  /**
   * Given the question and context, find the best answers.
   * @param question: string
   * @param context: string, context where the answers are looked up from.
   * @return array of answsers
   */
  findAnswers(question: string, context: string) {
    const features =
        this.process(question, context, MAX_QUERY_LEN, MAX_SEQ_LEN);
    const answers = features.map((feature, index) => {
      console.log(feature);
      const inputIds = tf.tensor2d(feature.inputIds, [1, size], 'int32');
      const segmentIds = tf.tensor2d(feature.segmentIds, [1, size], 'int32');
      const inputMask = tf.tensor2d(feature.inputMask, [1, size], 'int32');
      const globalStep = tf.scalar(index, 'int32');
      const result = this.model.execute(
          {
            input_ids: inputIds,
            segment_ids: segmentIds,
            input_mask: inputMask,
            global_step: globalStep
          },
          ['start_logits', 'end_logits']);
      const startLogits = result[0].arraySync();
      const endLogits = result[1].arraySync();
      return this.getBestAnswers(
          startLogits[0], endLogits[0], feature.origTokens,
          feature.tokenToOrigMap, index);
    });
    console.log(answers);
    return []
        .concat.apply([], answers)
        .sort((a, b) => b[1] - a[1])
        .slice(0, PREDICT_ANS_NUM);
  }

  /**
   * Find the Best N answers & logits from the logits array and input feature.
   * @param startLogits: start index for the answers
   * @param endLogits: end index for the answers
   * @param origTokens: original tokens of the passage
   * @param tokenToOrigMap: token to index mapping
   */
  getBestAnswers(
      startLogits: number[], endLogits: number[], origTokens: string[],
      tokenToOrigMap: {[key: string]: number}, docIndex = 0) {
    // Model uses the closed interval [start, end] for indices.
    const startIndexes = this.getBestIndex(startLogits);
    const endIndexes = this.getBestIndex(endLogits);

    const origResults = [];
    startIndexes.forEach(start => {
      endIndexes.forEach(end => {
        if (tokenToOrigMap[start] && tokenToOrigMap[end] && end >= start) {
          const length = end - start + 1;
          if (length < MAX_ANS_LEN) {
            origResults.push([start, end, startLogits[start] + endLogits[end]]);
          }
        }
      });
    });

    origResults.sort((a, b) => b[2] - a[2]);

    const answers = [];
    for (let i = 0; i < origResults.length; i++) {
      if (i >= PREDICT_ANS_NUM) {
        break;
      }

      let convertedText = '';
      if (origResults[i][0] > 0) {
        convertedText = this.convertBack(
            origTokens, tokenToOrigMap, origResults[i][0], origResults[i][1]);
      } else {
        convertedText = '';
      }
      answers.push([convertedText, origResults[i][2], docIndex]);
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  getBestIndex(logits: number[]) {
    const tmpList = [];
    for (let i = 0; i < MAX_SEQ_LEN; i++) {
      tmpList.push([i, i, logits[i]]);
    }
    tmpList.sort((a, b) => b[2] - a[2]);

    const indexes = [];
    for (let i = 0; i < PREDICT_ANS_NUM; i++) {
      indexes.push(tmpList[i][0]);
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  convertBack(
      origTokens: string[], tokenToOrigMap: {[key: string]: number},
      start: number, end: number) {
    // Shifted index is: index of logits + offset.
    const shiftedStart = start + OUTPUT_OFFSET;
    const shiftedEnd = end + OUTPUT_OFFSET;
    const startIndex = tokenToOrigMap[shiftedStart];
    const endIndex = tokenToOrigMap[shiftedEnd];
    // end + 1 for the closed interval.
    const ans = origTokens.slice(startIndex, endIndex + 1).join(' ');
    return ans;
  }
}

export const load = async(): Promise<MobileBert> => {
  const mobileBert = new MobileBertImpl();
  await mobileBert.load();
  return mobileBert;
};
