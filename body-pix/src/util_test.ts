/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
 *
 * =============================================================================
 */

// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {toValidInternalResolutionNumber} from './util';

describeWithFlags('util.toValidInternalResolutionNumber', ALL_ENVS, () => {
  it('produces correct output when small is specified', () => {
    const result = toValidInternalResolutionNumber('low');
    expect(result).toBe(257);
  });

  it('produces correct output when medium is specified', () => {
    const result = toValidInternalResolutionNumber('medium');
    expect(result).toBe(513);
  });

  it('produces correct output when large is specified', () => {
    const result = toValidInternalResolutionNumber('high');
    expect(result).toBe(1025);
  });

  it('produces correct output when number is specified', () => {
    for (let i = 0; i < 2000; i++) {
      const result = toValidInternalResolutionNumber(i);
      if (i < 161) {
        expect(result).toBe(161);
      }

      if (i > 1217) {
        expect(result).toBe(1217);
      }

      if (i === 250) {
        expect(result).toBe(257);
      }

      if (i === 500) {
        expect(result).toBe(513);
      }

      if (i === 750) {
        expect(result).toBe(737);
      }

      if (i === 1000) {
        expect(result).toBe(993);
      }
    }
  });
});
