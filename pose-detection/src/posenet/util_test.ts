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

import {assertValidResolution, toValidInputResolution} from './util';

describe('util', () => {
  describe('toValidInputResolution', () => {
    it('returns an odd value', () => {
      expect(toValidInputResolution(1920, 8) % 2).toEqual(1);
      expect(toValidInputResolution(1280, 16) % 2).toEqual(1);
      expect(toValidInputResolution(719, 16) % 2).toEqual(1);
      expect(toValidInputResolution(545, 16) % 2).toEqual(1);
      expect(toValidInputResolution(225, 8) % 2).toEqual(1);
      expect(toValidInputResolution(240, 8) % 2).toEqual(1);
    });

    it('returns the original value when already a valid resolution', () => {
      const outputStride = 16;

      const validResolution = toValidInputResolution(1000, outputStride);

      const resolution = toValidInputResolution(validResolution, outputStride);

      expect(resolution).toEqual(validResolution);
    });

    it('succeeds when 1-resolution is divisible by the output stride', () => {
      const outputStride = 8;
      const inputResolution = 562;

      const resolution = toValidInputResolution(inputResolution, outputStride);

      expect((resolution - 1) % outputStride).toEqual(0);
    });
  });

  describe('assertValidResolution', () => {
    it('throws when 1 - resolution is not divisible by output stride', () => {
      expect(() => {
        assertValidResolution(
            [
              toValidInputResolution(16 * 5, 16) + 1,
              toValidInputResolution(16 * 5, 16)
            ],
            16);
      }).toThrow();
    });
    it('doesnt throw when 1-resolution is divisible by output stride', () => {
      expect(() => {
        assertValidResolution(
            [
              toValidInputResolution(16 * 10, 16),
              toValidInputResolution(16 * 5 + 20, 16)
            ],
            16);
      }).not.toThrow();
      expect(() => {
        assertValidResolution(
            [
              toValidInputResolution(20 * 5, 32),
              toValidInputResolution(5 * 30 + 20, 32)
            ],
            32);
      }).not.toThrow();
    });
  });
});
