// /**
//  * @license
//  * Copyright 2019 Google Inc. All Rights Reserved.
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  *
//  =============================================================================
//  */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as bodyPixModel from './body_pix_model';
import * as resnet from './resnet';

describeWithFlags('BodyPix', NODE_ENVS, () => {
  let bodyPix: bodyPixModel.BodyPix;
  const inputResolution = 513;
  const outputStride = 32;
  const quantBytes = 4;
  const numKeypoints = 17;
  const numParts = 24;
  const outputResolution = (inputResolution - 1) / outputStride + 1;

  beforeAll((done) => {
    const resNetConfig = {
      architecture: 'ResNet50',
      outputStride: outputStride,
      inputResolution: inputResolution,
      quantBytes: quantBytes
    } as bodyPixModel.ModelConfig;

    spyOn(tfconv, 'loadGraphModel').and.callFake((): tfconv.GraphModel => {
      return null;
    })

    spyOn(resnet, 'ResNet').and.callFake(() => {
      return {
        outputStride,
        predict: (input: tf.Tensor3D) => {
          return {
            inputResolution,
            heatmapScores:
                tf.zeros([outputResolution, outputResolution, numKeypoints]),
            offsets: tf.zeros(
                [outputResolution, outputResolution, 2 * numKeypoints]),
            displacementFwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)]),
            displacementBwd: tf.zeros(
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)]),
            segmentation: tf.zeros([outputResolution, outputResolution, 1]),
            partHeatmaps:
                tf.zeros([outputResolution, outputResolution, numParts]),
            longOffsets: tf.zeros(
                [outputResolution, outputResolution, 2 * numKeypoints]),
            partOffsets:
                tf.zeros([outputResolution, outputResolution, 2 * numParts])
          };
        },
        dipose: () => {}
      };
    });

    bodyPixModel.load(resNetConfig)
        .then((bodyPixInstance: bodyPixModel.BodyPix) => {
          bodyPix = bodyPixInstance;
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimatePersonSegmentation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    bodyPix.estimateSinglePersonSegmentation(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimatePartSegmenation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;
    const beforeTensors = tf.memory().numTensors;
    bodyPix.estimateSinglePersonPartSegmentation(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });
});
