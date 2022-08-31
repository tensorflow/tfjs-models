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

import {ImageSize, Padding} from '../../shared/calculators/interfaces/common_interfaces';
import {InputResolution, Pose} from '../../types';

export function scalePoses(
    poses: Pose[], imageSize: ImageSize, inputResolution: InputResolution,
    padding: Padding): Pose[] {
  const {height, width} = imageSize;
  const scaleY =
      height / (inputResolution.height * (1 - padding.top - padding.bottom));
  const scaleX =
      width / (inputResolution.width * (1 - padding.left - padding.right));

  const offsetY = -padding.top * inputResolution.height;
  const offsetX = -padding.left * inputResolution.width;

  if (scaleX === 1 && scaleY === 1 && offsetY === 0 && offsetX === 0) {
    return poses;
  }

  for (const pose of poses) {
    for (const kp of pose.keypoints) {
      kp.x = (kp.x + offsetX) * scaleX;
      kp.y = (kp.y + offsetY) * scaleY;
    }
  }

  return poses;
}
