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

import cv from './assets/opencv';
import {connectedComponents} from './connectedComponents';

describe('connectedComponents', () => {
  it('The connectedComponents output matches with OpenCV.js on random input.',
     () => {
       const side = 50;
       const input =
           tf.tidy(() => ((tf.randomNormal([side, side]).sign()).add(1)).div(2))
               .toInt();
       const inputArray = input.arraySync() as number[][];
       const inputMatrix = cv.matFromArray(
           side, side, cv.CV_8U, [].concat.apply([], inputArray));

       // 0. The label counts match
       // 1. The dimensions match
       // 2. Labels match
       const conditions: boolean[] = [];

       const {labelsCount, labels} = connectedComponents(inputArray);
       const cvLabelsMatrix = new cv.Mat();
       const cvLabelsCount =
           cv.connectedComponents(inputMatrix, cvLabelsMatrix, 4);
       const height = cvLabelsMatrix.rows;
       const width = cvLabelsMatrix.cols;
       const cvLabels = Array.from(new Array(height), () => new Array(width));
       for (let rowIdx = 0; rowIdx < height; rowIdx++) {
         for (let colIdx = 0; colIdx < width; colIdx++) {
           const cvLabel = cvLabelsMatrix.ucharPtr(rowIdx, colIdx)[0];
           cvLabels[rowIdx][colIdx] = cvLabel;
         }
       }
       conditions.push(cvLabelsCount === labelsCount);
       const areDimsOk = height === width && height === side;
       conditions.push(areDimsOk);
       let areLabelsOk = true;
       for (let rowIdx = 0; rowIdx < height; rowIdx++) {
         for (let colIdx = 0; colIdx < width; colIdx++) {
           const cvLabel = cvLabels[rowIdx][colIdx];
           const label = labels[rowIdx][colIdx];
           areLabelsOk = cvLabel === label;
         }
       }
       conditions.push(areLabelsOk);
       cvLabelsMatrix.delete();
       expect(conditions)
           .toEqual(Array.from(new Array(conditions.length), () => true));
     });
});
