/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export const CV_8U: number;
export class Mat {
  readonly rows: number;
  readonly cols: number;
  readonly type: number;
  readonly channels: number;
  readonly depth: number;
  readonly dims: number;
  readonly empty: boolean;
  readonly step: number;
  readonly elemSize: number;
  readonly sizes: number[];
  constructor();
  delete(): void;
  ucharPtr(row: number, column: number): number[];
}

export class MatVector {
  get(i: number): MatVector;
  push_back(mat: Mat): void;
  delete(): void;
}

export declare function connectedComponents(
    graph: Mat, labels: Mat, connectivity: number): any;

declare function matFromArray<T>(
    rows: number, columns: number, type: number, array: number[]): Mat;

declare function cv(): void;
