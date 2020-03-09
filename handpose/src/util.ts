/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

export type TransformationMatrix = [
  [number, number, number], [number, number, number], [number, number, number]
];

export function normalizeRadians(angle: number) {
  return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
}

export function computeRotation(
    point1: [number, number], point2: [number, number]) {
  const radians =
      Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
  return normalizeRadians(radians);
}

const buildTranslationMatrix = (x: number, y: number): TransformationMatrix =>
    ([[1, 0, x], [0, 1, y], [0, 0, 1]]);

export function dot(v1: number[], v2: number[]) {
  let product = 0;
  for (let i = 0; i < v1.length; i++) {
    product += v1[i] * v2[i];
  }
  return product;
}

export function getColumnFrom2DArr(
    arr: number[][], columnIndex: number): number[] {
  const column: number[] = [];

  for (let i = 0; i < arr.length; i++) {
    column.push(arr[i][columnIndex]);
  }

  return column;
}

function multiplyTransformMatrices(
    mat1: number[][], mat2: number[][]): TransformationMatrix {
  const product = [];

  const size = mat1.length;

  for (let row = 0; row < size; row++) {
    product.push([]);
    for (let col = 0; col < size; col++) {
      product[row].push(dot(mat1[row], getColumnFrom2DArr(mat2, col)));
    }
  }

  return product as TransformationMatrix;
}

export function buildRotationMatrix(
    rotation: number, center: [number, number]): TransformationMatrix {
  const cosA = Math.cos(rotation);
  const sinA = Math.sin(rotation);

  const rotationMatrix = [[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]];
  const translationMatrix = buildTranslationMatrix(center[0], center[1]);
  const translationTimesRotation =
      multiplyTransformMatrices(translationMatrix, rotationMatrix);

  const negativeTranslationMatrix =
      buildTranslationMatrix(-center[0], -center[1]);
  return multiplyTransformMatrices(
      translationTimesRotation, negativeTranslationMatrix);
}

export function invertTransformMatrix(matrix: TransformationMatrix):
    TransformationMatrix {
  const rotationComponent =
      [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]];
  const translationComponent = [matrix[0][2], matrix[1][2]];
  const invertedTranslation = [
    -dot(rotationComponent[0], translationComponent),
    -dot(rotationComponent[1], translationComponent)
  ];

  return [
    rotationComponent[0].concat(
        invertedTranslation[0]) as [number, number, number],
    rotationComponent[1].concat(
        invertedTranslation[1]) as [number, number, number],
    [0, 0, 1]
  ];
}

export function rotatePoint(
    homogeneousCoordinate: [number, number, number],
    rotationMatrix: TransformationMatrix): [number, number] {
  return [
    dot(homogeneousCoordinate, rotationMatrix[0]),
    dot(homogeneousCoordinate, rotationMatrix[1])
  ];
}
