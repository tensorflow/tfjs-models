/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {calculateAssociationNormRect} from './association_norm_rect';
import {Rect} from './interfaces/shape_interfaces';

//  0.4                                         ================
//                                              |    |    |    |
//  0.3 =====================                   |   NR2   |    |
//      |    |    |   NR1   |                   |    |    NR4  |
//  0.2 |   NR0   |    ===========              ================
//      |    |    |    |    |    |
//  0.1 =====|===============    |
//           |    NR3  |    |    |
//  0.0      ================    |
//                     |   NR5   |
// -0.1                ===========
//     0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2

// NormalizedRect nr0.
const nr0: Rect = {
  xCenter: 0.2,
  yCenter: 0.2,
  width: 0.2,
  height: 0.2
};

// NormalizedRect nr1.
const nr1: Rect = {
  xCenter: 0.4,
  yCenter: 0.2,
  width: 0.2,
  height: 0.2
};

// NormalizedRect nr2.
const nr2: Rect = {
  xCenter: 1.0,
  yCenter: 0.3,
  width: 0.2,
  height: 0.2
};

// NormalizedRect nr3.
const nr3: Rect = {
  xCenter: 0.35,
  yCenter: 0.15,
  width: 0.3,
  height: 0.3
};

// NormalizedRect nr4.
const nr4: Rect = {
  xCenter: 1.1,
  yCenter: 0.3,
  width: 0.2,
  height: 0.2
};

// NormalizedRect nr5.
const nr5: Rect = {
  xCenter: 0.45,
  yCenter: 0.05,
  width: 0.3,
  height: 0.3
};

describe('calculateAssociationNormRect', () => {
  it('3 inputs.', async () => {
    const minSimilarityThreshold = 0.1;
    const inputList0 = [nr0, nr1, nr2];
    const inputList1 = [nr3, nr4];
    const inputList2 = [nr5];

    const result = calculateAssociationNormRect(
        [inputList0, inputList1, inputList2], minSimilarityThreshold);

    // nr3 overlaps with nr0, nr1 and nr5 overlaps with nr3. Since nr5 is
    // in the highest priority, we remove other rects.
    // nr4 overlaps with nr2, and nr4 is higher priority, so we keep it.
    // The final output therefore contains 2 elements.
    expect(result.length).toBe(2);
    // Outputs are in order of inputs, so nr4 is before nr5 in output vector.

    // det_4 overlaps with det_2.
    expect(result[0]).toBe(nr4);

    // det_3 overlaps with det_0.
    // det_3 overlaps with det_1.
    // det_5 overlaps with det_3.
    expect(result[1]).toBe(nr5);
  });

  it('3 inputs reverse.', async () => {
    const minSimilarityThreshold = 0.1;
    const inputList0 = [nr5];
    const inputList1 = [nr3, nr4];
    const inputList2 = [nr0, nr1, nr2];

    const result = calculateAssociationNormRect(
        [inputList0, inputList1, inputList2], minSimilarityThreshold);

    // nr3 overlaps with nr5, so nr5 is removed. nr0 overlaps with nr3, so
    // nr3 is removed as nr0 is in higher priority for keeping. nr2 overlaps
    // with nr4 so nr4 is removed as nr2 is higher priority for keeping.
    // The final output therefore contains 3 elements.
    expect(result.length).toBe(3);
    // Outputs are in order of inputs, so nr4 is before nr5 in output vector.

    // Outputs are in same order as inputs.
    expect(result[0]).toBe(nr0);
    expect(result[1]).toBe(nr1);
    expect(result[2]).toBe(nr2);
  });

  it('single input.', async () => {
    const minSimilarityThreshold = 0.1;
    const inputList0 = [nr3, nr5];

    const result =
        calculateAssociationNormRect([inputList0], minSimilarityThreshold);

    // nr5 overlaps with nr3. Since nr5 is after nr3 in the same input
    // stream we remove nr3 and keep nr5. The final output therefore contains
    // 1 elements.
    expect(result.length).toBe(1);
    expect(result[0]).toBe(nr5);
  });
});
