/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {assertValidResolution, getValidInputResolution, getValidResolution} from './util';

describe('util', () => {
  describe('getValidResolution', () => {
    it('returns an odd value', () => {
      expect(getValidResolution(0.5, 545, 32) % 2).toEqual(1);
      expect(getValidResolution(0.5, 545, 16) % 2).toEqual(1);
      expect(getValidResolution(0.5, 545, 8) % 2).toEqual(1);
      expect(getValidResolution(0.845, 242, 8) % 2).toEqual(1);
      expect(getValidResolution(0.421, 546, 16) % 2).toEqual(1);
    });

    it('returns a value that when 1 is subtracted by it is ' +
           'divisible by the output stride',
       () => {
         const outputStride = 32;
         const imageSize = 562;

         const scaleFactor = 0.63;

         const resolution =
             getValidResolution(scaleFactor, imageSize, outputStride);

         expect((resolution - 1) % outputStride).toEqual(0);
       });
  });

  describe('getValidInputResolution', () => {
    it('returns an odd value', () => {
      expect(getValidInputResolution(1920, 8) % 2).toEqual(1);
      expect(getValidInputResolution(1280, 16) % 2).toEqual(1);
      expect(getValidInputResolution(719, 16) % 2).toEqual(1);
      expect(getValidInputResolution(545, 16) % 2).toEqual(1);
      expect(getValidInputResolution(225, 8) % 2).toEqual(1);
      expect(getValidInputResolution(240, 8) % 2).toEqual(1);
    });

    it('returns a value that when 1 is subtracted by it is ' +
           'divisible by the output stride',
       () => {
         const outputStride = 8;
         const inputResolution = 562;

         const resolution =
             getValidInputResolution(inputResolution, outputStride);

         expect((resolution - 1) % outputStride).toEqual(0);
       });
  });

  describe('assertValidResolution', () => {
    it('raises an error when one subtracted by the input resolution is ' +
           'not divisible by the output stride',
       () => {
         expect(() => {
           assertValidResolution(16 * 5, 16);
         }).toThrow();
         expect(() => {
           assertValidResolution(8 * 10, 8);
         }).toThrow();
         expect(() => {
           assertValidResolution(32 * 10 + 5, 32);
         }).toThrow();
       });
    it('does not raise an error when one subtracted by the input resolution is ' +
           'divisible by the output stride',
       () => {
         expect(() => {
           assertValidResolution(16 * 5 + 1, 16);
         }).not.toThrow();
         expect(() => {
           assertValidResolution(32 * 10 + 1, 32);
         }).not.toThrow();
       });
  });
});
