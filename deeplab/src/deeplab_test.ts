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
import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS,} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {load} from '.';

describeWithFlags('SemanticSegmentation', NODE_ENVS, () => {
  it('SemanticSegmentation should not leak', async () => {
    const model = await load();
    const x: tf.Tensor3D = tf.zeros([227, 500, 3]);
    const numOfTensorsBefore = tf.memory().numTensors;

    await model.segment(x);
    expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
  });

  it('SemanticSegmentation map has matching dimensions', async () => {
    const x: tf.Tensor3D = tf.zeros([513, 500, 3]);
    const model = await load();
    const segmentationMapTensor = await model.predict(x);
    const [height, width] = segmentationMapTensor.shape;
    expect([height, width]).toEqual([513, 500]);
  });

  it('SemanticSegmentation segment method generates valid output', async () => {
    const model = await load();
    const x: tf.Tensor3D = tf.zeros([300, 500, 3]);

    const {legend} = await model.segment(x);
    expect(Object.keys(legend)).toContain('background');
  });

});
