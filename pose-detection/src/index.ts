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
// Top level exports.
// Entry point to create a new detector instance.
export {BlazePoseMediaPipeEstimationConfig, BlazePoseMediaPipeModelConfig, BlazePoseModelType} from './blazepose_mediapipe/types';
export {BlazePoseTfjsEstimationConfig, BlazePoseTfjsModelConfig} from './blazepose_tfjs/types';
export {createDetector} from './create_detector';
export {MoveNetEstimationConfig, MoveNetModelConfig} from './movenet/types';
// PoseDetector class.
export {PoseDetector} from './pose_detector';
export {PoseNetEstimationConfig, PosenetModelConfig} from './posenet/types';

// Supported models enum.
export * from './types';

export {TrackerType} from './calculators/types';

// Second level exports.
// Utils for rendering.
import * as util from './util';
export {util};

// General calculators.
import {keypointsToNormalizedKeypoints} from './shared/calculators/keypoints_to_normalized_keypoints';
const calculators = {keypointsToNormalizedKeypoints};
export {calculators};

// MoveNet model types.
import {SINGLEPOSE_LIGHTNING, SINGLEPOSE_THUNDER, MULTIPOSE_LIGHTNING} from './movenet/constants';
const movenet = {
  modelType: {
    'SINGLEPOSE_LIGHTNING': SINGLEPOSE_LIGHTNING,
    'SINGLEPOSE_THUNDER': SINGLEPOSE_THUNDER,
    'MULTIPOSE_LIGHTNING': MULTIPOSE_LIGHTNING
  }
};
export {movenet};
