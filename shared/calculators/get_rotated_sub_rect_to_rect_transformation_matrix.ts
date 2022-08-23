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

import {arrayToMatrix4x4, Matrix4x4} from './calculate_inverse_matrix';
import {Rect} from './interfaces/shape_interfaces';

/**
 * Generates a 4x4 projective transform matrix M, so that for any point in the
 * subRect image p(x, y), we can use the matrix to calculate the projected point
 * in the original image p' (x', y'): p' = p * M;
 *
 * @param subRect Rotated sub rect in absolute coordinates.
 * @param rectWidth
 * @param rectHeight
 * @param flipHorizontaly Whether to flip the image horizontally.
 */
// Ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tensor/image_to_tensor_utils.h
export function getRotatedSubRectToRectTransformMatrix(
    subRect: Rect, rectWidth: number, rectHeight: number,
    flipHorizontally: boolean): Matrix4x4 {
  // The resulting matrix is multiplication of below commented out matrices:
  //   postScaleMatrix
  //     * translateMatrix
  //     * rotateMatrix
  //     * flipMatrix
  //     * scaleMatrix
  //     * initialTranslateMatrix

  // For any point in the transformed image p, we can use the above matrix to
  // calculate the projected point in the original image p'. So that:
  // p' = p * M;
  // Note: The transform matrix below assumes image coordinates is normalized
  // to [0, 1] range.

  // Matrix to convert X,Y to [-0.5, 0.5] range "initialTranslateMatrix"
  // [ 1.0,  0.0, 0.0, -0.5]
  // [ 0.0,  1.0, 0.0, -0.5]
  // [ 0.0,  0.0, 1.0,  0.0]
  // [ 0.0,  0.0, 0.0,  1.0]

  const a = subRect.width;
  const b = subRect.height;
  // Matrix to scale X,Y,Z to sub rect "scaleMatrix"
  // Z has the same scale as X.
  // [   a, 0.0, 0.0, 0.0]
  // [0.0,    b, 0.0, 0.0]
  // [0.0, 0.0,    a, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const flip = flipHorizontally ? -1 : 1;
  // Matrix for optional horizontal flip around middle of output image.
  // [ fl  , 0.0, 0.0, 0.0]
  // [ 0.0, 1.0, 0.0, 0.0]
  // [ 0.0, 0.0, 1.0, 0.0]
  // [ 0.0, 0.0, 0.0, 1.0]

  const c = Math.cos(subRect.rotation);
  const d = Math.sin(subRect.rotation);
  // Matrix to do rotation around Z axis "rotateMatrix"
  // [    c,   -d, 0.0, 0.0]
  // [    d,    c, 0.0, 0.0]
  // [ 0.0, 0.0, 1.0, 0.0]
  // [ 0.0, 0.0, 0.0, 1.0]

  const e = subRect.xCenter;
  const f = subRect.yCenter;
  // Matrix to do X,Y translation of sub rect within parent rect
  // "translateMatrix"
  // [1.0, 0.0, 0.0, e   ]
  // [0.0, 1.0, 0.0, f   ]
  // [0.0, 0.0, 1.0, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const g = 1.0 / rectWidth;
  const h = 1.0 / rectHeight;
  // Matrix to scale X,Y,Z to [0.0, 1.0] range "postScaleMatrix"
  // [g,    0.0, 0.0, 0.0]
  // [0.0, h,    0.0, 0.0]
  // [0.0, 0.0,    g, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const matrix: number[] = new Array(16);
  // row 1
  matrix[0] = a * c * flip * g;
  matrix[1] = -b * d * g;
  matrix[2] = 0.0;
  matrix[3] = (-0.5 * a * c * flip + 0.5 * b * d + e) * g;

  // row 2
  matrix[4] = a * d * flip * h;
  matrix[5] = b * c * h;
  matrix[6] = 0.0;
  matrix[7] = (-0.5 * b * c - 0.5 * a * d * flip + f) * h;

  // row 3
  matrix[8] = 0.0;
  matrix[9] = 0.0;
  matrix[10] = a * g;
  matrix[11] = 0.0;

  // row 4
  matrix[12] = 0.0;
  matrix[13] = 0.0;
  matrix[14] = 0.0;
  matrix[15] = 1.0;

  return arrayToMatrix4x4(matrix);
}
