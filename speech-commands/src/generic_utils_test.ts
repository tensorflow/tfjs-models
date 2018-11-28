
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {arrayBuffer2String, string2ArrayBuffer} from './generic_utils';

describe('string2ArrayBuffer and arrayBuffer2String', () => {
  it('round trip: ASCII only', () => {
    const str = 'Lorem_Ipsum_123 !@#$%^&*()';
    expect(arrayBuffer2String(string2ArrayBuffer(str))).toEqual(str);
  });
  it('round trip: non-ASCII', () => {
    const str = 'Welcome 欢迎 स्वागत हे ようこそ добро пожаловать 😀😀';
    expect(arrayBuffer2String(string2ArrayBuffer(str))).toEqual(str);
  });
  it('round trip: empty string', () => {
    const str = '';
    expect(arrayBuffer2String(string2ArrayBuffer(str))).toEqual(str);
  });
});
