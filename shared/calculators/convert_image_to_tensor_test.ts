/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {loadImage} from '../test_util';
import {convertImageToTensor} from './convert_image_to_tensor';
import {transformValueRange} from './image_utils';
import {ImageToTensorConfig} from './interfaces/config_interfaces';
import {Rect} from './interfaces/shape_interfaces';
import {toImageDataLossy} from './mask_util';

// Measured in pixels.
const EPS = 7;

describe('ImageToTensorCalculator', () => {
  let input: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    await tf.ready();
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    input = await loadImage('shared/image_to_tensor_input.jpg', 64, 128);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  async function runTest(
      roi: Rect, config: ImageToTensorConfig, expectedFileName: string) {
    const {imageTensor} = convertImageToTensor(input, config, roi);

    // Shift to [0, 255] range.
    const [rangeMin, rangeMax] = config.outputTensorFloatRange;
    const {scale, offset} = transformValueRange(rangeMin, rangeMax, 0, 255);
    const actualResult =
        tf.tidy(() => tf.add(tf.mul(imageTensor, scale), offset));
    tf.dispose(imageTensor);

    const expectedResultRGBA =
        await loadImage(
            `shared/${expectedFileName}`, config.outputTensorSize.width,
            config.outputTensorSize.height)
            .then(image => toImageDataLossy(image));
    const expectedResultRGB =
        expectedResultRGBA.data.filter((_, index) => index % 4 !== 3);

    expectArraysClose(
        new Uint8ClampedArray(actualResult.dataSync()), expectedResultRGB, EPS);
  }

  it('medium sub rect keep aspect.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: true,
          borderMode: 'replicate'
        },
        'medium_sub_rect_keep_aspect.png',
    );
  });

  it('medium sub rect keep aspect border zero.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: true,
          borderMode: 'zero'
        },

        'medium_sub_rect_keep_aspect_border_zero.png',
    );
  });

  it('medium sub rect keep aspect with rotation.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: Math.PI * 90 / 180,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: true,
          borderMode: 'replicate'
        },

        'medium_sub_rect_keep_aspect_with_rotation.png',
    );
  });

  it('medium sub rect keep aspect with rotation border zero.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: Math.PI * 90 / 180,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: true,
          borderMode: 'zero'
        },

        'medium_sub_rect_keep_aspect_with_rotation_border_zero.png',
    );
  });

  it('medium sub rect with rotation.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: Math.PI * -45 / 180,
        },
        {
          outputTensorFloatRange: [-1, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: false,
          borderMode: 'replicate'
        },
        'medium_sub_rect_with_rotation.png',
    );
  });

  it('medium sub rect with rotation border zero.', async () => {
    await runTest(
        {
          xCenter: 0.65,
          yCenter: 0.4,
          width: 0.5,
          height: 0.5,
          rotation: Math.PI * -45 / 180,
        },
        {
          outputTensorFloatRange: [-1, 1],
          outputTensorSize: {width: 256, height: 256},
          keepAspectRatio: false,
          borderMode: 'zero'
        },

        'medium_sub_rect_with_rotation_border_zero.png',
    );
  });

  it('large sub rect.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: false,
          borderMode: 'replicate'
        },
        'large_sub_rect.png',
    );
  });

  it('large sub rect border zero.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: false,
          borderMode: 'zero'
        },
        'large_sub_rect_border_zero.png',
    );
  });

  it('large sub keep aspect.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: true,
          borderMode: 'replicate'
        },
        'large_sub_rect_keep_aspect.png',
    );
  });

  it('large sub keep aspect border zero.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: true,
          borderMode: 'zero'
        },
        'large_sub_rect_keep_aspect_border_zero.png',
    );
  });

  it('large sub keep aspect with rotation.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: Math.PI * -15 / 180,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: true,
          borderMode: 'replicate'
        },
        'large_sub_rect_keep_aspect_with_rotation.png',
    );
  });

  it('large sub keep aspect with rotation border zero.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.5,
          height: 1.1,
          rotation: Math.PI * -15 / 180,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 128, height: 128},
          keepAspectRatio: true,
          borderMode: 'zero'
        },
        'large_sub_rect_keep_aspect_with_rotation_border_zero.png');
  });

  it('no op except range.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.0,
          height: 1.0,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 64, height: 128},
          keepAspectRatio: true,
          borderMode: 'replicate'
        },
        'noop_except_range.png',
    );
  });

  it('no op except range border zero.', async () => {
    await runTest(
        {
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1.0,
          height: 1.0,
          rotation: 0,
        },
        {
          outputTensorFloatRange: [0, 1],
          outputTensorSize: {width: 64, height: 128},
          keepAspectRatio: true,
          borderMode: 'zero'
        },
        'noop_except_range.png',
    );
  });
});
