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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import * as mobilenet from './mobilenet';
import * as posenetModel from './posenet_model';
import * as resnet from './resnet';

describeWithFlags('PoseNet', NODE_ENVS, () => {
  let mobileNet: posenetModel.PoseNet;
  let resNet: posenetModel.PoseNet;
  const inputResolution = 513;
  const outputStride = 32;
  const multiplier = 1.0;
  const quantBytes = 4;
  const outputResolution = (inputResolution - 1) / outputStride + 1;
  const numKeypoints = 17;

  beforeAll(async () => {
    // Mock out the actual load so we don't make network requests in the unit
    // test.
    const resNetConfig =
        {architecture: 'ResNet50', outputStride, inputResolution, quantBytes} as
        posenetModel.ModelConfig;

    const mobileNetConfig = {
      architecture: 'MobileNetV1',
      outputStride,
      inputResolution,
      multiplier,
      quantBytes
    } as posenetModel.ModelConfig;

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
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)])
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
                [outputResolution, outputResolution, 2 * (numKeypoints - 1)])
          };
        },
        dipose: () => {}
      };
    });

    resNet = await posenetModel.load(resNetConfig);
    mobileNet = await posenetModel.load(mobileNetConfig);
  });

  it('estimateSinglePose does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([inputResolution, inputResolution, 3]);

    const beforeTensors = tf.memory().numTensors;

    await resNet.estimateSinglePose(input, {flipHorizontal: false});
    await mobileNet.estimateSinglePose(input, {flipHorizontal: false});

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('estimateMultiplePoses does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([inputResolution, inputResolution, 3]);

    const beforeTensors = tf.memory().numTensors;
    await resNet.estimateMultiplePoses(input, {
      flipHorizontal: false,
      maxDetections: 5,
      scoreThreshold: 0.5,
      nmsRadius: 20
    });

    await mobileNet.estimateMultiplePoses(input, {
      flipHorizontal: false,
      maxDetections: 5,
      scoreThreshold: 0.5,
      nmsRadius: 20
    });

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });
});
