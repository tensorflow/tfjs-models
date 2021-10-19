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

export type Coord2D = [number, number];
export type Coord3D = [number, number, number];
export type Coords3D = Coord3D[];

export type TransformationMatrix = [
  [number, number, number], [number, number, number], [number, number, number]
];

export const IDENTITY_MATRIX: TransformationMatrix =
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/**
 * Normalizes the provided angle to the range -pi to pi.
 * @param angle The angle in radians to be normalized.
 */
export function normalizeRadians(angle: number): number {
  return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
}

/**
 * Computes the angle of rotation between two anchor points.
 * @param point1 First anchor point
 * @param point2 Second anchor point
 */
export function computeRotation(
    point1: Coord2D|Coord3D, point2: Coord2D|Coord3D): number {
  const radians =
      Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
  return normalizeRadians(radians);
}

export function radToDegrees(rad: number): number {
  return rad * 180 / Math.PI;
}

function buildTranslationMatrix(x: number, y: number): TransformationMatrix {
  return [[1, 0, x], [0, 1, y], [0, 0, 1]];
}

export function dot(v1: number[], v2: number[]): number {
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
    rotation: number, center: Coord2D): TransformationMatrix {
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
    rotationComponent[0].concat(invertedTranslation[0]) as Coord3D,
    rotationComponent[1].concat(invertedTranslation[1]) as Coord3D, [0, 0, 1]
  ];
}

export function rotatePoint(
    homogeneousCoordinate: Coord3D,
    rotationMatrix: TransformationMatrix): Coord2D {
  return [
    dot(homogeneousCoordinate, rotationMatrix[0]),
    dot(homogeneousCoordinate, rotationMatrix[1])
  ];
}

export function xyDistanceBetweenPoints(
    a: Coord2D|Coord3D, b: Coord2D|Coord3D): number {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function normalizeVector(v: Coord3D): Coord3D {
    const l = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
    return [v[0] / l, v[1] / l, v[2] / l];
}

function crossP(v1: Coord3D, v2: Coord3D): Coord3D {
    const e1 = v1[1] * v2[2] - v1[2] * v2[1];
    const e2 = v1[2] * v2[0] - v1[0] * v2[2];
    const e3 = v1[0] * v2[1] - v1[1] * v2[0];

    return [e1, e2, e3];
}

 /**
  * 
  * @param meshPoints - points produced by facemesh model
  * @returns normalized direction vectors x, y, z; right-handed system
  */
export function getDirectionVectors(meshPoints: Coords3D): [Coord3D, Coord3D, Coord3D] {
    // These points lie *almost* on same line in uv coordinates
    const p1 = meshPoints[127]; // cheek
    const p2 = meshPoints[356]; // cheek
    const p3 = meshPoints[6]; // nose

    const vhelp = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]] as Coord3D;
    const vx_d = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]] as Coord3D;
    const vy_d = crossP(vhelp, vx_d);

    const vx = normalizeVector(vx_d);
    const vy = normalizeVector(vy_d);
    const vz = normalizeVector(crossP(vx_d, vy_d));

    return [vx, vy, vz];
 }

 /**
  * 
  * @param meshPoints - points produced by facemesh model
  * @returns 3x3 rotation matrix; right-handed system
  */
export function getRotationMatrix(meshPoints: Coords3D): TransformationMatrix {
    const [vx, vy, vz] = getDirectionVectors();

    return [
        [vx[0], vy[0], vz[0]],
        [vx[1], vy[1], vz[1]],
        [vx[2], vy[2], vz[2]]
    ];

 }
