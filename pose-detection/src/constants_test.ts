/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
// tslint:disable-next-line: no-imports-from-dist
import {expectArraysEqual} from '@tensorflow/tfjs-core/dist/test_util';

import {BLAZEPOSE_KEYPOINTS, BLAZEPOSE_KEYPOINTS_BY_NAME, COCO_KEYPOINTS, COCO_KEYPOINTS_BY_NAME} from './constants';

describe('Keypoint names are consistent ', () => {
  it('Coco.', () => {
    expectArraysEqual(COCO_KEYPOINTS, Object.keys(COCO_KEYPOINTS_BY_NAME));
  });

  it('BlazePose.', () => {
    expectArraysEqual(
        BLAZEPOSE_KEYPOINTS, Object.keys(BLAZEPOSE_KEYPOINTS_BY_NAME));
  });
});
