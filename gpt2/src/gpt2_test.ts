/**
 * @license
 * Copyright 2023 Google LLC.
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

import {GPT2} from './gpt2';

describe('gpt2', () => {
  let gpt2: GPT2;
  beforeEach(() => {
    gpt2 = new GPT2();
  });

  it('this is a fake test', async () => {
    expect(await gpt2.generate('asdf')).toEqual(' the park');
  });
});
