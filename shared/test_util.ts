/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

/** Karma server directory serving local files. */
export const KARMA_SERVER = './base/test_data';

export async function loadImage(
    imagePath: string, width: number,
    height: number): Promise<HTMLImageElement> {
  const img = new Image(width, height);
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    img.crossOrigin = '';
    img.onload = () => {
      resolve(img);
    };
  });

  img.src = `${KARMA_SERVER}/${imagePath}`;

  return promise;
}

export function imageToBooleanMask(
    rgbaData: Uint8ClampedArray, r: number, g: number, b: number) {
  const mask: boolean[] = [];
  for (let i = 0; i < rgbaData.length; i += 4) {
    mask.push(rgbaData[i] >= r && rgbaData[i + 1] >= g && rgbaData[i + 2] >= b);
  }
  return mask;
}

export function segmentationIOU(
    expectedMask: boolean[], actualMask: boolean[]) {
  expect(expectedMask.length === actualMask.length);

  const sum = (mask: boolean[]) => mask.reduce((a, b) => a + +b, 0);

  const intersectionMask =
      expectedMask.map((value, index) => value && actualMask[index]);
  const iou = sum(intersectionMask) /
      (sum(expectedMask) + sum(actualMask) - sum(intersectionMask) +
       Number.EPSILON);

  return iou;
}
