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
import * as tf from '@tensorflow/tfjs';
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {Tokenizer} from './tokenizer';

describeWithFlags('Universal Sentence Encoder', tf.test_util.NODE_ENVS, () => {
  let tokenizer: Tokenizer;
  beforeAll(() => {
    tokenizer = new Tokenizer([
      ['�', 0], ['<s>', 0], ['</s>', 0], ['extra_token_id_1', 0],
      ['extra_token_id_2', 0], ['extra_token_id_3', 0], ['a', -1], ['ç', -2]
    ]);
  });

  it('should normalize inputs', () => {
    expect(tokenizer.encode('ça')).toEqual(tokenizer.encode('c\u0327a'));
  });

  it('should handle unknown inputs', () => {
    expect(tokenizer.encode('😹')).not.toThrow();
  });

  it('should treat contiguous unknown inputs as a single word', () => {
    expect(tokenizer.encode('a😹😹')).toEqual([6, 0]);
  });
});
