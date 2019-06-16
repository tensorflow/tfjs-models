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

import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {BodyPix, load, mobilenetLoader} from './body_pix_model';

describeWithFlags('BodyPix', NODE_ENVS, () => {
  let net: BodyPix;

  beforeAll((done) => {
    // Mock out the actual load so we don't make network requests in the unit
    // test.
    spyOn(mobilenetLoader, 'load').and.callFake(() => {
      return {
        predict: () => tf.zeros([1000]),
        convToOutput:
            (mobileNetOutput: tf.Tensor3D, outputLayerName: string) => {
              const shapes: {[layer: string]: number[]} = {
                'segment_2': [23, 17, 1],
                'part_heatmap_2': [23, 17, 24]
              };
              return tf.zeros(shapes[outputLayerName]);
            }
      };
    });

    load()
        .then((model: BodyPix) => {
          net = model;
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimatePersonSegmentation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    net.estimatePersonSegmentation(input, 16)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });
  it('estimatePartSegmenation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;
    const beforeTensors = tf.memory().numTensors;
    net.estimatePartSegmentation(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });
});
