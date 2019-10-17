/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
 *
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as bodyPixModel from './body_pix_model';
import * as resnet from './resnet';
import {toValidInternalResolutionNumber} from './util';

describeWithFlags('BodyPix', NODE_ENVS, () => {
  let bodyPix: bodyPixModel.BodyPix;
  const inputResolution = 513;
  const outputStride = 32;
  const quantBytes = 4;
  const numKeypoints = 17;
  const numParts = 24;
  const outputResolution = (inputResolution - 1) / outputStride + 1;

  beforeAll((done) => {
    const resNetConfig =
        {architecture: 'ResNet50', outputStride, inputResolution, quantBytes} as
        bodyPixModel.ModelConfig;

    spyOn(tfconv, 'loadGraphModel').and.callFake((): tfconv.GraphModel => {
      return null;
    });

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

  it('segmentPerson does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    bodyPix.segmentPerson(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimatePersonPartSegmenation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;
    const beforeTensors = tf.memory().numTensors;
    bodyPix.segmentPersonParts(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('toValidInternalResolutionNumber produces correct output when small is specified',
     () => {
       const result = toValidInternalResolutionNumber('small');
       expect(result).toBe(257);
     })

  it('toValidInternalResolutionNumber produces correct output when medium is specified',
     () => {
       const result = toValidInternalResolutionNumber('medium');
       expect(result).toBe(513);
     })

  it('toValidInternalResolutionNumber produces correct output when large is specified',
     () => {
       const result = toValidInternalResolutionNumber('large');
       expect(result).toBe(1025);
     })

  it('toValidInternalResolutionNumber produces correct output when number is specified',
     () => {
       for (let i = 0; i < 2000; i++) {
         const result = toValidInternalResolutionNumber(i);
         if (i < 161) {
           expect(result).toBe(161);
         }

         if (i > 1217) {
           expect(result).toBe(1217);
         }

         if (i == 250) {
           expect(result).toBe(257);
         }

         if (i == 500) {
           expect(result).toBe(513);
         }

         if (i == 750) {
           expect(result).toBe(737);
         }

         if (i == 1000) {
           expect(result).toBe(993);
         }
       }
     })
});
