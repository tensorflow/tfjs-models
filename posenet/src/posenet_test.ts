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

import * as tf from '@tensorflow/tfjs';
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as posenetModel from './posenet_model';

describeWithFlags('PoseNet', tf.test_util.NODE_ENVS, () => {
  let net: posenetModel.PoseNet;

  beforeAll((done) => {
    // Mock out the actual load so we don't make network requests in the unit
    // test.
    spyOn(posenetModel.mobilenetLoader, 'load').and.callFake(() => {
      return {
        predict: () => tf.zeros([1000]),
        convToOutput:
            (mobileNetOutput: tf.Tensor3D, outputLayerName: string) => {
              const shapes: {[layer: string]: number[]} = {
                'heatmap_2': [16, 16, 17],
                'offset_2': [16, 16, 34],
                'displacement_fwd_2': [16, 16, 32],
                'displacement_bwd_2': [16, 16, 32]
              };
              return tf.zeros(shapes[outputLayerName]);
            }
      };
    });

    posenetModel.load()
        .then((posenetInstance: posenetModel.PoseNet) => {
          net = posenetInstance;
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimateSinglePose does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    net.estimateSinglePose(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimateMultiplePoses does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;
    net.estimateMultiplePoses(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });
});
