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

import * as bodySegmentation from '../index';
import {Mask} from '../shared/calculators/interfaces/common_interfaces';
import {toImageDataLossy, toTensorLossy} from '../shared/calculators/mask_util';
import * as renderUtil from '../shared/calculators/render_util';
import {loadImage} from '../shared/test_util';
import {BodyPixSegmentationConfig} from './types';

// Measured in channels.
const DIFF_IMAGE = 30;

class CanvasImageSourceMask implements Mask {
  constructor(private mask: CanvasImageSource) {}

  async toCanvasImageSource() {
    return this.mask;
  }

  async toImageData() {
    return toImageDataLossy(this.mask);
  }

  async toTensor() {
    return toTensorLossy(this.mask);
  }

  getUnderlyingType() {
    return 'canvasimagesource' as const ;
  }
}

async function getSegmentation(
    image: HTMLImageElement, config: BodyPixSegmentationConfig) {
  const segmenter = await bodySegmentation.createSegmenter(
      bodySegmentation.SupportedModels.BodyPix);

  const segmentations = await segmenter.segmentPeople(image, config);
  return Promise.all(segmentations.map(async segmentation => {
    return {
      maskValueToLabel: segmentation.maskValueToLabel,
      // Convert to canvas image source to apply alpha-premultiplication.
      mask: new CanvasImageSourceMask(
          await segmentation.mask.toCanvasImageSource())
    };
  }));
}

async function getBinaryMask(
    image: HTMLImageElement, expectedNumSegmentations?: number) {
  const segmentation = await getSegmentation(image, {
    multiSegmentation: expectedNumSegmentations != null,
    segmentBodyParts: false
  });

  if (expectedNumSegmentations != null) {
    expect(segmentation.length).toBe(expectedNumSegmentations);
  }

  const binaryMask = await renderUtil.toBinaryMask(
      segmentation, {r: 255, g: 255, b: 255, a: 255},
      {r: 0, g: 0, b: 0, a: 255});
  return binaryMask;
}

async function getColoredMask(
    image: HTMLImageElement, expectedNumSegmentations?: number) {
  const segmentation = await getSegmentation(image, {
    multiSegmentation: expectedNumSegmentations != null,
    segmentBodyParts: true
  });

  if (expectedNumSegmentations != null) {
    expect(segmentation.length).toBe(expectedNumSegmentations);
  }

  const coloredMask = await renderUtil.toColoredMask(
      segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor,
      {r: 255, g: 255, b: 255, a: 255});

  return coloredMask;
}

const WIDTH = 1049;
const HEIGHT = 861;

async function expectImage(actual: ImageData, imageName: string) {
  const expectedImage = await loadImage(imageName, WIDTH, HEIGHT)
                            .then(async image => toImageDataLossy(image));
  const mismatchedChannels = actual.data.reduce(
      (mismatched, channel, i) =>
          mismatched + +(channel !== expectedImage.data[i]),
      0);
  expect(mismatchedChannels).toBeLessThanOrEqual(DIFF_IMAGE);
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

  it('Single Segmentation + No body parts.', async () => {
    const binaryMask = await getBinaryMask(image);
    await expectImage(binaryMask, 'shared/three_people_binary_mask.png');
  });

  it('Multi Segmentation + No body parts.', async () => {
    const binaryMask = await getBinaryMask(image, 3);
    await expectImage(binaryMask, 'shared/three_people_binary_mask.png');
  });

  it('Single Segmentation + Body parts.', async () => {
    const coloredMask = await getColoredMask(image);
    await expectImage(coloredMask, 'shared/three_people_colored_mask.png');
  });

  it('Multi Segmentation + Body parts.', async () => {
    const coloredMask = await getColoredMask(image, 3);
    await expectImage(coloredMask, 'shared/three_people_colored_mask.png');
  });
});
