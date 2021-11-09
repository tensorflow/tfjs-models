import {Keypoint} from '../../shared/calculators/interfaces/common_interfaces';
import {BoundingBox} from '../../shared/calculators/interfaces/shape_interfaces';

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

export interface Track {
  id: number;              // A unique identifier for each tracked person.
  lastTimestamp: number;   // The last timestamp (in milliseconds) in which a
                           // detection was linked with the track.
  keypoints?: Keypoint[];  // Keypoints associated with the tracked person.
  box?: BoundingBox;       // Bounding box associated with the tracked person.
  score?: number;          // A confidence value of the track.
}
