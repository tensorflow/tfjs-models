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
import {Keypoint} from '../types';

// Estimate object scale to use its inverse value as velocity scale for
// RelativeVelocityFilter. If value will be too small (less than config's
// `minAllowedObjectScale`) smoothing will be disabled and landmarks will be
// returned as is.
// Object scale is calculated as average between bounding box width and height
// with sides parallel to axis.
// TODO(lina128): Follow up with bazarevsky@ about a better pose scale approach.
export function getObjectScale(landmarks: Keypoint[]): number {
  const x = landmarks.map(landmark => landmark.x);
  const xMin = Math.min(...x);
  const xMax = Math.max(...x);

  const y = landmarks.map(landmark => landmark.y);
  const yMin = Math.min(...y);
  const yMax = Math.max(...y);

  const objectWidth = xMax - xMin;
  const objectHeight = yMax - yMin;

  return (objectWidth + objectHeight) / 2;
}
