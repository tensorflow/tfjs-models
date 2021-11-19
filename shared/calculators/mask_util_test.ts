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

import {tensor3d} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysEqual} from '@tensorflow/tfjs-core/dist/test_util';

import * as maskUtil from './mask_util';

describe('maskUtil', () => {
  const rgbaValues: number[] = Array.from(new Array(24).keys());
  const expectedExact = rgbaValues;
  const expectedLossy = [
    0,  0,  0,  3,  0,  0,  0,  7,  0,  0,  0,  11,
    17, 17, 17, 15, 13, 13, 13, 19, 22, 22, 22, 23
  ];
  it('ImageData to HTMLCanvasElement.', async () => {
    const imageData = new ImageData(new Uint8ClampedArray(rgbaValues), 2, 3);
    const canvas = await maskUtil.toHTMLCanvasElementLossy(imageData);

    expect(canvas.width).toBe(imageData.width);
    expect(canvas.height).toBe(imageData.height);

    const actual =
        Array.from(canvas.getContext('2d')
                       .getImageData(0, 0, canvas.width, canvas.height)
                       .data);
    expectArraysEqual(actual, expectedLossy);
  });

  it('Tensor to HTMLCanvasElement.', async () => {
    const tensor = tensor3d(rgbaValues, [2, 3, 4], 'int32');
    const [height, width] = tensor.shape.slice(0, 2);
    const canvas = await maskUtil.toHTMLCanvasElementLossy(tensor);

    expect(canvas.width).toBe(width);
    expect(canvas.height).toBe(height);

    const actual =
        Array.from(canvas.getContext('2d')
                       .getImageData(0, 0, canvas.width, canvas.height)
                       .data);
    expectArraysEqual(actual, expectedLossy);
  });

  it('HTMLCanvasElement to ImageData.', async () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = 2;
    canvas.height = 3;
    ctx.putImageData(
        new ImageData(new Uint8ClampedArray(rgbaValues), 2, 3), 0, 0);

    const imageData = await maskUtil.toImageDataLossy(canvas);

    expect(imageData.width).toBe(canvas.width);
    expect(imageData.height).toBe(canvas.height);

    const actual = Array.from(imageData.data);
    expectArraysEqual(actual, expectedLossy);
  });

  it('Tensor to ImageData.', async () => {
    const tensor = tensor3d(rgbaValues, [2, 3, 4], 'int32');
    const [height, width] = tensor.shape.slice(0, 2);
    const imageData = await maskUtil.toImageDataLossy(tensor);

    expect(imageData.width).toBe(width);
    expect(imageData.height).toBe(height);

    const actual = Array.from(imageData.data);
    expectArraysEqual(actual, expectedExact);
  });

  it('HTMLCanvasElement to Tensor.', async () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = 2;
    canvas.height = 3;
    ctx.putImageData(
        new ImageData(new Uint8ClampedArray(rgbaValues), 2, 3), 0, 0);

    const tensor = await maskUtil.toTensorLossy(canvas);
    const [height, width] = tensor.shape.slice(0, 2);

    expect(width).toBe(canvas.width);
    expect(height).toBe(canvas.height);

    expectArraysEqual(tensor.dataSync(), expectedLossy);
  });

  it('ImageData to Tensor.', async () => {
    const imageData = new ImageData(new Uint8ClampedArray(rgbaValues), 2, 3);

    const tensor = await maskUtil.toTensorLossy(imageData);
    const [height, width] = tensor.shape.slice(0, 2);

    expect(width).toBe(imageData.width);
    expect(height).toBe(imageData.height);

    expectArraysEqual(tensor.dataSync(), expectedExact);
  });

  it('assertMaskValue.', async () => {
    expect(() => maskUtil.assertMaskValue(-1))
        .toThrowError(/Mask value must be in range \[0, 255\]/);
    expect(() => maskUtil.assertMaskValue(256))
        .toThrowError(/Mask value must be in range \[0, 255\]/);
    expect(() => maskUtil.assertMaskValue(1.1))
        .toThrowError(/must be an integer/);

    const uint8Values = Array.from(Array(256).keys());
    uint8Values.forEach(
        value => expect(() => maskUtil.assertMaskValue(value)).not.toThrow());
  });
});
