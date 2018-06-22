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

import {getValidResolution} from './util';

describe('util.getValidResolution', () => {
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
