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
import {describeWithFlags, NODE_ENVS,} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {DummyModel} from '.';

describeWithFlags('Dummy', NODE_ENVS, () => {
  it('DummyModel detect method should generate no output', async () => {
    const dummy = new DummyModel();
    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;

    const data = await dummy.predict(x);

    expect(data).toEqual();
  });
});
