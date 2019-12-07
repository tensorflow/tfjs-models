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

const BASE_DIR = 'mobile-bert/';
const MODEL_URL = BASE_DIR + 'model.json';
const size = 384;
let tokenizer = {};
const globalStep = tf.scalar(1, 'int32');
const MAX_ANS_LEN = 32;
// const MAX_QUERY_LEN = 64;
const MAX_SEQ_LEN = 384;
const PREDICT_ANS_NUM = 5;
const OUTPUT_OFFSET = 1;
export const convert =
    (tokenizer: BertTokenizer, query: string, context: string,
     maxQueryLen: number, maxSeqLen: number) => {
      let queryTokens = tokenizer.tokenize(query);
      if (queryTokens.length > maxQueryLen) {
        queryTokens = queryTokens.slice(0, maxQueryLen);
      }
      const origTokens = context.trim().split(/\s+/).slice(0);
      const tokenToOrigIndex = ([]);
      let allDocTokens = ([]);
      for (let i = 0; i < origTokens.length; i++) {
        const token = origTokens[i];
        const subTokens = tokenizer.tokenize(token);
        for (let j = 0; j < subTokens.length; j++) {
          const subToken = subTokens[j];
          tokenToOrigIndex.push(i);
          allDocTokens.push(subToken);
        }
      }
      const maxContextLen = maxSeqLen - queryTokens.length - 3;
      if (allDocTokens.length > maxContextLen) {
        allDocTokens = allDocTokens.slice(0, maxContextLen);
      }
      const tokens = [];
      const segmentIds = [];
      const tokenToOrigMap = ({});
      tokens.push(tokenizer.CLS_INDEX);
      segmentIds.push(0);
      for (let i = 0; i < queryTokens.length; i++) {
        const queryToken = queryTokens[i];
        tokens.push(queryToken);
        segmentIds.push(0);
      }
      tokens.push(tokenizer.SEP_INDEX);
      segmentIds.push(0);
      for (let i = 0; i < allDocTokens.length; i++) {
        const docToken = allDocTokens[i];
        tokens.push(docToken);
        segmentIds.push(1);
        tokenToOrigMap[tokens.length] = tokenToOrigIndex[i];
      }
      tokens.push(tokenizer.SEP_INDEX);
      segmentIds.push(1);
      const inputIds = tokens;
      const inputMask = inputIds.map(id => 1);
      while ((inputIds.length < maxSeqLen)) {
        inputIds.push(0);
        inputMask.push(0);
        segmentIds.push(0);
      }
      return {inputIds, inputMask, segmentIds, origTokens, tokenToOrigMap};
    };

export const load = async () => {
  const bertModel = await tfconv.loadGraphModel(MODEL_URL);
  // warm up the webgl
  const inputIds = tf.ones([1, size], 'int32');
  const segmentIds = tf.ones([1, size], 'int32');
  const inputMask = tf.ones([1, size], 'int32');

  bertModel.predict({
    input_ids: inputIds,
    segment_ids: segmentIds,
    input_mask: inputMask,
    global_step: globalStep
  });

  tokenizer = await loadTokenizer();
  return {bertModel, tokenizer};
};

/**
 * Find the Best N answers & logits from the logits array and input feature.
 * @param startLogits: start index for the answers
 * @param endLogits: end index for the answers
 * @param origTokens: original tokens of the passage
 * @param tokenToOrigMap: token to index mapping
 */
export const getBestAnswers =
    (startLogits: number[], endLogits: number[], origTokens: string[],
     tokenToOrigMap: {[key: string]: number}) => {
      // Model uses the closed interval [start, end] for indices.
      const startIndexes = getBestIndex(startLogits);
      const endIndexes = getBestIndex(endLogits);

      const origResults = [];
      startIndexes.forEach(start => {
        endIndexes.forEach(end => {
          if (tokenToOrigMap[start] && tokenToOrigMap[end] && end >= start) {
            const length = end - start + 1;
            if (length < MAX_ANS_LEN) {
              origResults.push(
                  [start, end, startLogits[start] + endLogits[end]]);
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
          convertedText = convertBack(
              origTokens, tokenToOrigMap, origResults[i][0], origResults[i][1]);
        } else {
          convertedText = '';
        }
        answers.push(convertedText);
      }
      return answers;
    };

/** Get the n-best logits from a list of all the logits. */
const getBestIndex =
    (logits: number[]) => {
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
const convertBack =
    (origTokens: string[], tokenToOrigMap: {[key: string]: number},
     start: number, end: number) => {
      // Shifted index is: index of logits + offset.
      const shiftedStart = start + OUTPUT_OFFSET;
      const shiftedEnd = end + OUTPUT_OFFSET;
      const startIndex = tokenToOrigMap[shiftedStart];
      const endIndex = tokenToOrigMap[shiftedEnd];
      // end + 1 for the closed interval.
      const ans = origTokens.slice(startIndex, endIndex + 1).join(' ');
      return ans;
    };
