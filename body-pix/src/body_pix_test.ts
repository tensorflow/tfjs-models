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
import * as mobilenet from './mobilenet';
import * as resnet from './resnet';
import {toValidInputResolution} from './util';

describeWithFlags('BodyPix', NODE_ENVS, () => {
  let mobileNet: bodyPixModel.BodyPix;
  let resNet: bodyPixModel.BodyPix;
  const inputResolution = 513;
  const outputStride = 32;
  const multiplier = 1.0;
  const quantBytes = 4;
  const numKeypoints = 17;
  const numParts = 24;
  const outputResolution = (inputResolution - 1) / outputStride + 1;

  beforeAll((done) => {
    const resNetConfig =
        {architecture: 'ResNet50', outputStride, inputResolution, quantBytes} as
        bodyPixModel.ModelConfig;

    const mobileNetConfig = {
      architecture: 'MobileNetV1',
      outputStride,
      inputResolution,
      multiplier,
      quantBytes
    } as bodyPixModel.ModelConfig;

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

    spyOn(mobilenet, 'MobileNet').and.callFake(() => {
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
          resNet = bodyPixInstance;
        })
        .then(() => bodyPixModel.load(mobileNetConfig))
        .then((bodyPixInstance: bodyPixModel.BodyPix) => {
          mobileNet = bodyPixInstance;
        })
        .then(done)
        .catch(done.fail);
  });

  it('segmentPerson does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;

    const beforeTensors = tf.memory().numTensors;

    resNet.segmentPerson(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('estimatePersonPartSegmenation does not leak memory', done => {
    const input = tf.zeros([513, 513, 3]) as tf.Tensor3D;
    const beforeTensors = tf.memory().numTensors;
    mobileNet.segmentPersonParts(input)
        .then(() => {
          expect(tf.memory().numTensors).toEqual(beforeTensors);
        })
        .then(done)
        .catch(done.fail);
  });

  it('load with mobilenet when input resolution is a number returns a model ' +
         'with a valid and the same input resolution width and height',
     (done) => {
       const inputResolution = 500;
       const validInputResolution =
           toValidInputResolution(inputResolution, outputStride);

       const expectedResolution = [validInputResolution, validInputResolution];

       bodyPixModel
           .load({architecture: 'MobileNetV1', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with resnet when input resolution is a number returns a model ' +
         'with a valid and the same input resolution width and height',
     (done) => {
       const inputResolution = 350;
       const validInputResolution =
           toValidInputResolution(inputResolution, outputStride);

       const expectedResolution = [validInputResolution, validInputResolution];

       bodyPixModel
           .load({architecture: 'ResNet50', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with mobilenet when input resolution is an object with width and height ' +
         'returns a model with a valid resolution for the width and height',
     (done) => {
       const inputResolution = {width: 600, height: 400};

       const expectedResolution = [
         toValidInputResolution(inputResolution.height, outputStride),
         toValidInputResolution(inputResolution.width, outputStride)
       ];

       bodyPixModel
           .load({architecture: 'MobileNetV1', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });

  it('load with resnet when input resolution is an object with width and height ' +
         'returns a model with a valid resolution for the width and height',
     (done) => {
       const inputResolution = {width: 700, height: 500};

       const expectedResolution = [
         toValidInputResolution(inputResolution.height, outputStride),
         toValidInputResolution(inputResolution.width, outputStride)
       ];

       bodyPixModel
           .load({architecture: 'ResNet50', outputStride, inputResolution})
           .then(model => {
             expect(model.inputResolution).toEqual(expectedResolution);

             done();
           });
     });
});
