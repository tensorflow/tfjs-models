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
import {BlazePoseMediaPipeEstimationConfig} from './blazepose_mediapipe/types';
import {BlazePoseTfjsEstimationConfig} from './blazepose_tfjs/types';
import {MoveNetEstimationConfig} from './movenet/types';
import {PoseNetEstimationConfig} from './posenet/types';
import {Pose, PoseDetectorInput} from './types';

/**
 * User-facing interface for all pose detectors.
 */
export interface PoseDetector {
  /**
   * Estimate poses for an image or video frame.
   * @param image An image or video frame.
   * @param config Optional. See `EstimationConfig` for available options.
   * @param timestamp Optional. In milliseconds. This is useful when image is
   *     a tensor, which doesn't have timestamp info. Or to override timestamp
   *     in a video.
   * @returns An array of poses, each pose contains an array of `Keypoint`s.
   */
  estimatePoses(
      image: PoseDetectorInput,
      config?: PoseNetEstimationConfig|BlazePoseTfjsEstimationConfig|
      BlazePoseMediaPipeEstimationConfig|MoveNetEstimationConfig,
      timestamp?: number): Promise<Pose[]>;

  /**
   * Dispose the underlying models from memory.
   */
  dispose(): void;

  /**
   * Reset global states in the model.
   */
  reset(): void;
}
