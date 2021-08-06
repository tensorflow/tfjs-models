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
import {Rect} from './interfaces/shape_interfaces';

/**
 * Estimate object scale to allow filter work similarly on nearer or futher
 * objects.
 * @param roi Non-normalized rectangle.
 * @returns A number representing the object scale.
 */
export function getObjectScale(roi: Rect): number {
  const objectWidth = roi.width;
  const objectHeight = roi.height;

  return (objectWidth + objectHeight) / 2;
}
