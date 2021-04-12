/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {RobustOneEuroFilter} from './robust_one_euro_filter';

describeWithFlags('MoveNet', ALL_ENVS, () => {
  let robust_filter: RobustOneEuroFilter;
  let timeout: number;

  beforeAll(() => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 300000;  // 5mins
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  beforeEach(async () => {
    robust_filter = new RobustOneEuroFilter();
  });

  it('robust filter produces right output shape', async () => {
    const input: number[] = [-1.0, 2.0, 3.0];

    const output: number[] = robust_filter.insert(input);

    expect(input.length).toEqual(output.length);
  });

  it ('robust filter outputs are in convex hull of inputs', async() => {
    const input_0: number[] = [-1.0, 2.0, 3.0];
    const input_1: number[] = [5.0, -2.5, 7.0];

    robust_filter.insert(input_0);
    const output_1: number[] = robust_filter.insert(input_1);

    expect(output_1[0]).toBeLessThan(5.0);
    expect(output_1[0]).toBeGreaterThan(-1.0);

    expect(output_1[1]).toBeLessThan(2.0);
    expect(output_1[1]).toBeGreaterThan(-2.5);

    expect(output_1[2]).toBeLessThan(7.0);
    expect(output_1[2]).toBeGreaterThan(3.0);

  });
});
