/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs';

import {DepthPredict} from './index';

describe('DepthPredict', () => {
  it('should load the model with no parameters', async () => {
    const depthprediction = new DepthPredict();
    await depthprediction.load();
    const img: tf.Tensor3D = tf.zeros([480, 640, 3])
    const out = depthprediction.predict(img);
    expect(out.shape).toEqual([224, 224, 1]);
  });
  it('should be able to output raw tensors', async () => {
    const depthprediction = new DepthPredict(0,255,true);
    await depthprediction.load();
    const img: tf.Tensor3D = tf.zeros([480, 640, 3]);
    const out = depthprediction.predict(img);
    expect(out.shape).toEqual([1, 224, 224]);
  });
});
