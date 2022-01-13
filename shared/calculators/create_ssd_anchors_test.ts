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
// tslint:disable-next-line: no-imports-from-dist
import {expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';

import {KARMA_SERVER} from '../test_util';

import {createSsdAnchors} from './create_ssd_anchors';
import {AnchorConfig} from './interfaces/config_interfaces';
import {Rect} from './interfaces/shape_interfaces';

const EPS = 1e-5;

function compareAnchors(actual: Rect[], expected: Rect[]) {
  expect(actual.length).toBe(expected.length);

  for (let i = 0; i < actual.length; ++i) {
    const actualAnchor = actual[i];
    const expectedAnchor = expected[i];

    for (const key of ['xCenter', 'yCenter', 'width', 'height'] as const ) {
      const actualValue = actualAnchor[key];
      const expectedValue = expectedAnchor[key];

      expectNumbersClose(actualValue, expectedValue, EPS);
    }
  }
}

function parseAnchors(anchorsValues: number[][]): Rect[] {
  return anchorsValues.map(anchorValues => ({
                             xCenter: anchorValues[0],
                             yCenter: anchorValues[1],
                             width: anchorValues[2],
                             height: anchorValues[3]
                           }));
}

describe('createSsdAnchors', () => {
  it('face detection config.', async () => {
    const expectedAnchors = parseAnchors(
        await fetch(`${KARMA_SERVER}/shared/anchor_golden_file_0.json`)
            .then(response => response.json()));
    const config: AnchorConfig = {
      featureMapHeight: [] as number[],
      featureMapWidth: [] as number[],
      numLayers: 5,
      minScale: 0.1171875,
      maxScale: 0.75,
      inputSizeHeight: 256,
      inputSizeWidth: 256,
      anchorOffsetX: 0.5,
      anchorOffsetY: 0.5,
      strides: [8, 16, 32, 32, 32],
      aspectRatios: [1.0],
      fixedAnchorSize: true
    };
    const actualAnchors = createSsdAnchors(config);

    compareAnchors(actualAnchors, expectedAnchors);
  });

  it('3 inputs reverse.', async () => {
    const expectedAnchors = parseAnchors(
        await fetch(`${KARMA_SERVER}/shared/anchor_golden_file_1.json`)
            .then(response => response.json()));

    const config: AnchorConfig = {
      featureMapHeight: [] as number[],
      featureMapWidth: [] as number[],
      numLayers: 6,
      minScale: 0.2,
      maxScale: 0.95,
      inputSizeHeight: 300,
      inputSizeWidth: 300,
      anchorOffsetX: 0.5,
      anchorOffsetY: 0.5,
      strides: [16, 32, 64, 128, 256, 512],
      aspectRatios: [1.0, 2.0, 0.5, 3.0, 0.3333],
      reduceBoxesInLowestLayer: true
    };
    const actualAnchors = createSsdAnchors(config);

    compareAnchors(actualAnchors, expectedAnchors);
  });
});
