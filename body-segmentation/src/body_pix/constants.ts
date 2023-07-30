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
import {Color} from '../shared/calculators/interfaces/common_interfaces';
import {assertMaskValue} from '../shared/calculators/mask_util';

const RAINBOW_PART_COLORS: Array<[number, number, number]> = [
  [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
  [238, 67, 149], [255, 78, 125], [255, 94, 99],  [255, 115, 75],
  [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
  [175, 240, 91], [135, 245, 87], [96, 247, 96],  [64, 243, 115],
  [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
  [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
];

export function bodyPixMaskValueToRainbowColor(maskValue: number): Color {
  assertMaskValue(maskValue);
  if (maskValue < RAINBOW_PART_COLORS.length) {
    const [r, g, b] = RAINBOW_PART_COLORS[maskValue];
    return {r, g, b, a: 255};
  }
  throw new Error(
      `Mask value must be in range [0, ${RAINBOW_PART_COLORS.length})`);
}
