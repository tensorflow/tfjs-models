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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as bodyPixModel from './body_pix_model';
import * as resnet from './resnet';
import * as util from './util';

describeWithFlags('BodyPix', ALL_ENVS, () => {
  let bodyPix: bodyPixModel.BodyPix;
  const inputResolution = 513;
  const outputStride = 32;
  const quantBytes = 4;
  const numKeypoints = 17;
  const numParts = 24;
  const outputResolution = (inputResolution - 1) / outputStride + 1;

  beforeAll(async () => {
    const resNetConfig =
        {architecture: 'ResNet50', outputStride, inputResolution, quantBytes} as
        bodyPixModel.ModelConfig;

    spyOn(tfconv, 'loadGraphModel').and.callFake(() => {
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
        // tslint:disable-next-line:no-any
      } as any as resnet.ResNet;
    });

    bodyPix = await bodyPixModel.load(resNetConfig);
  });

  it('segmentPerson does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([73, 73, 3]);

    const beforeTensors = tf.memory().numTensors;

    await bodyPix.segmentPerson(input);
    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('segmentMultiPerson does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([73, 73, 3]);

    const beforeTensors = tf.memory().numTensors;

    await bodyPix.segmentMultiPerson(input);
    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('segmentPersonParts does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
    const beforeTensors = tf.memory().numTensors;
    await bodyPix.segmentPersonParts(input);
    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('segmentMultiPersonParts does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
    const beforeTensors = tf.memory().numTensors;
    await bodyPix.segmentMultiPersonParts(input);
    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it(`segmentPerson uses default values when null is ` +
         `passed in inferenceConfig parameters`,
     async () => {
       const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
       spyOn(util, 'toInputResolutionHeightAndWidth').and.callThrough();
       await bodyPix.segmentPerson(input, {});
       expect(util.toInputResolutionHeightAndWidth)
           .toHaveBeenCalledWith('medium', 32, [73, 73]);
     });

  it(`segmentMultiPerson uses default values when null is ` +
         `passed in inferenceConfig parameters`,
     async () => {
       const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
       spyOn(util, 'toInputResolutionHeightAndWidth').and.callThrough();
       await bodyPix.segmentMultiPerson(input, {});
       expect(util.toInputResolutionHeightAndWidth)
           .toHaveBeenCalledWith('medium', 32, [73, 73]);
     });

  it(`segmentPersonParts uses default values when null is ` +
         `passed in inferenceConfig parameters`,
     async () => {
       const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
       spyOn(util, 'toInputResolutionHeightAndWidth').and.callThrough();
       await bodyPix.segmentPersonParts(input, {});
       expect(util.toInputResolutionHeightAndWidth)
           .toHaveBeenCalledWith('medium', 32, [73, 73]);
     });

  it(`segmentMultiPersonParts uses default values when null is ` +
         `passed in inferenceConfig parameters`,
     async () => {
       const input: tf.Tensor3D = tf.zeros([73, 73, 3]);
       spyOn(util, 'toInputResolutionHeightAndWidth').and.callThrough();
       await bodyPix.segmentMultiPersonParts(input, {});
       expect(util.toInputResolutionHeightAndWidth)
           .toHaveBeenCalledWith('medium', 32, [73, 73]);
     });
});
