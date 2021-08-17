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
import {calculateLandmarkProjection} from './calculate_landmark_projection';

describe('calculateLandmarkProjection', () => {
  it('default rectangle.', async () => {
    const landmarks = [{x: 10, y: 20, z: -0.5}];
    const rect = {
      xCenter: 0.5,
      yCenter: 0.5,
      height: 1.0,
      width: 1.0,
      rotation: 0,
    };

    const result = calculateLandmarkProjection(landmarks, rect);

    expect(result[0].x).toBe(10);
    expect(result[0].y).toBe(20);
    expect(result[0].z).toBe(-0.5);
  });

  it('cropped rectangle.', async () => {
    const landmarks = [{x: 1, y: 1, z: -0.5}];
    const rect = {
      xCenter: 0.5,
      yCenter: 0.5,
      height: 2,
      width: 0.5,
      rotation: 0,
    };

    const result = calculateLandmarkProjection(landmarks, rect);

    expect(result[0].x).toBe(0.75);
    expect(result[0].y).toBe(1.5);
    expect(result[0].z).toBe(-0.25);
  });
});
