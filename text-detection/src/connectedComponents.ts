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

import {UnionFind} from './unionFind';

export function connectedComponents(graph: number[][]):
    {labelsCount: number, labels: number[][]} {
  const height = graph.length;
  const width = graph[0].length;
  const labels = Array.from(Array(height), () => new Array(width));
  const roster = new UnionFind();
  // make the label for the background
  roster.makeLabel();
  for (let row = 0; row < height; ++row) {
    for (let col = 0; col < width; ++col) {
      if (graph[row][col] > 0) {
        if (row > 0 && graph[row - 1][col] > 0) {
          if (col > 0 && graph[row][col - 1] > 0) {
            labels[row][col] =
                roster.union(labels[row][col - 1], labels[row - 1][col]);
          } else {
            labels[row][col] = labels[row - 1][col];
          }
        } else {
          if (col > 0 && graph[row][col - 1] > 0) {
            labels[row][col] = labels[row][col - 1];
          } else {
            labels[row][col] = roster.makeLabel();
          }
        }
      } else {
        labels[row][col] = 0;
      }
    }
  }
  const labelsCount = roster.flattenAndRelabel();
  for (let row = 0; row < height; ++row) {
    for (let col = 0; col < width; ++col) {
      const label = labels[row][col];
      labels[row][col] = roster.getComponent(label);
    }
  }
  return {labelsCount, labels};
}
