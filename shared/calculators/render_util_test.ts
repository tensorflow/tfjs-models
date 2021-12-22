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

// tslint:disable-next-line: no-imports-from-dist
import {BROWSER_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {loadImage} from '../test_util';
import {Color, Mask, Segmentation} from './interfaces/common_interfaces';

import {toHTMLCanvasElementLossy, toImageDataLossy, toTensorLossy} from './mask_util';
import * as renderUtil from './render_util';

class ImageDataMask implements Mask {
  constructor(private mask: ImageData) {}

  async toCanvasImageSource() {
    return toHTMLCanvasElementLossy(this.mask);
  }

  async toImageData() {
    return this.mask;
  }

  async toTensor() {
    return toTensorLossy(this.mask);
  }

  getUnderlyingType() {
    return 'imagedata' as const ;
  }
}

function getSegmentation(imageData: ImageData): Segmentation {
  return {
    maskValueToLabel: (maskValue: number) => `${maskValue}`,
    mask: new ImageDataMask(imageData),
  };
}

const WIDTH = 1049;
const HEIGHT = 861;

async function getBinarySegmentation() {
  const image = await loadImage(
      'shared/three_people_binary_segmentation.png', WIDTH, HEIGHT);
  const imageData = await toImageDataLossy(image);

  return [getSegmentation(imageData)];
}

async function getBinaryMask() {
  const segmentation = await getBinarySegmentation();
  const binaryMask = await renderUtil.toBinaryMask(
      segmentation, {r: 255, g: 255, b: 255, a: 255},
      {r: 0, g: 0, b: 0, a: 255});
  return binaryMask;
}

async function getColoredSegmentation() {
  const image = await loadImage(
      'shared/three_people_colored_segmentation.png', WIDTH, HEIGHT);
  const imageData = await toImageDataLossy(image);

  return [getSegmentation(imageData)];
}

const RAINBOW_PART_COLORS: Array<[number, number, number]> = [
  [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
  [238, 67, 149], [255, 78, 125], [255, 94, 99],  [255, 115, 75],
  [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
  [175, 240, 91], [135, 245, 87], [96, 247, 96],  [64, 243, 115],
  [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
  [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
];

function maskValueToRainbowColor(maskValue: number): Color {
  const [r, g, b] = RAINBOW_PART_COLORS[maskValue];
  return {r, g, b, a: 255};
}

async function getColoredMask() {
  const segmentation = await getColoredSegmentation();

  const coloredMask = await renderUtil.toColoredMask(
      segmentation, maskValueToRainbowColor, {r: 255, g: 255, b: 255, a: 255});

  return coloredMask;
}

async function expectImage(actual: ImageData, imageName: string) {
  loadImage(imageName, WIDTH, HEIGHT)
      .then(image => toImageDataLossy(image))
      .then(
          async expectedImage =>
              expectArraysClose(actual.data, expectedImage.data));
}

describeWithFlags('renderUtil', BROWSER_ENVS, () => {
  let image: HTMLImageElement;
  let timeout: number;

  beforeAll(async () => {
    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000;  // 2mins

    image = await loadImage('shared/three_people.jpg', WIDTH, HEIGHT);
  });

  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
  });

  it('toBinaryMask.', async () => {
    const binaryMask = await getBinaryMask();
    await expectImage(binaryMask, 'shared/three_people_binary_mask.png');
  });

  it('toColoredMask.', async () => {
    const coloredMask = await getColoredMask();
    await expectImage(coloredMask, 'shared/three_people_colored_mask.png');
  });

  it('drawMask.', async () => {
    const binaryMask = await getBinaryMask();

    const canvas = document.createElement('canvas');
    await renderUtil.drawMask(canvas, image, binaryMask, 0.7, 3);

    const imageMask = await toImageDataLossy(canvas);

    await expectImage(imageMask, 'shared/three_people_draw_mask.png');
  });

  it('drawPixelatedMask.', async () => {
    const coloredMask = await getColoredMask();

    const canvas = document.createElement('canvas');
    await renderUtil.drawPixelatedMask(canvas, image, coloredMask);

    const imageMask = await toImageDataLossy(canvas);

    await expectImage(imageMask, 'shared/three_people_pixelated_mask.png');
  });

  it('drawBokehEffect.', async () => {
    const binarySegmentation = await getBinarySegmentation();

    const canvas = document.createElement('canvas');
    await renderUtil.drawBokehEffect(canvas, image, binarySegmentation);

    const imageMask = await toImageDataLossy(canvas);

    await expectImage(imageMask, 'shared/three_people_bokeh_effect.png');
  });

  it('blurBodyPart.', async () => {
    const coloredSegmentation = await getColoredSegmentation();

    const canvas = document.createElement('canvas');
    await renderUtil.blurBodyPart(canvas, image, coloredSegmentation, [0, 1]);

    const imageMask = await toImageDataLossy(canvas);

    await expectImage(imageMask, 'shared/three_people_blur_body_parts.png');
  });
});
