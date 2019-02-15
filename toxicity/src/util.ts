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

// We pad inputs with 0's to limit the number of different shapes that the model
// uses. This enables texture reuse for performance.
export const padInputs = (inputs: number[][]): number[][] => {
  const longest = Math.max(...inputs.map(d => d.length));
  let nearestBucket = 8;
  while (nearestBucket < longest) {
    nearestBucket *= 2;
  }

  return inputs.map(
      d => {return d.concat(
          Array.from({length: nearestBucket - d.length}, () => 0))});
};