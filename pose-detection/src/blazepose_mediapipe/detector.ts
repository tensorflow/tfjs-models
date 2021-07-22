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
import * as pose from '@mediapipe/pose';
import {BLAZEPOSE_KEYPOINTS} from '../constants';

import {PoseDetector} from '../pose_detector';
import {NamedPoseMap, Pose, PoseDetectorInput} from '../types';
import {validateModelConfig} from './detector_utils';

import {BlazePoseMediaPipeEstimationConfig, BlazePoseMediaPipeModelConfig} from './types';

/**
 * MediaPipe detector class.
 */
class BlazePoseMediaPipeDetector implements PoseDetector {
  private readonly poseSolution: pose.Pose;

  // This will be filled out by asynchronous calls to onResults. They will be
  // stable after `await send` is called on the pose solution.
  private width = 0;
  private height = 0;
  private poses: Pose[];
  private poses3D: Pose[];

  private selfieMode = false;

  // Should not be called outside.
  constructor(config: BlazePoseMediaPipeModelConfig) {
    this.poseSolution = new pose.Pose({
      locateFile: (path, base) => {
        if (config.solutionPath) {
          const solutionPath = config.solutionPath.replace(/\/+$/, '');
          return `${solutionPath}/${path}`;
        }
        return `${base}/${path}`;
      }
    });
    let modelComplexity: 0|1|2;
    switch (config.modelType) {
      case 'lite':
        modelComplexity = 0;
        break;
      case 'heavy':
        modelComplexity = 2;
        break;
      case 'full':
      default:
        modelComplexity = 1;
        break;
    }
    this.poseSolution.setOptions({
      modelComplexity,
      smoothLandmarks: config.enableSmoothing || true,
      selfieMode: this.selfieMode,
    });
    this.poseSolution.onResults((results) => {
      this.height = results.image.height;
      this.width = results.image.width;
      this.poses =
          this.translateOutputs(results.poseLandmarks, true /* shouldScale */);
      this.poses3D = this.translateOutputs(
          results.poseWorldLandmarks, false /* shouldScale */);
    });
  }

  private translateOutputs(
      poses: pose.NormalizedLandmarkList|pose.LandmarkList,
      shouldScale: boolean): Pose[] {
    if (poses == null) {
      return [];
    }

    if (shouldScale) {
      return [{
        keypoints: poses.map((landmark, i) => ({
                               x: landmark.x * this.width,
                               y: landmark.y * this.height,
                               z: landmark.z,
                               score: landmark.visibility,
                               name: BLAZEPOSE_KEYPOINTS[i]
                             }))
      }];
    } else {
      return [{
        keypoints: poses.map((landmark, i) => ({
                               x: landmark.x,
                               y: landmark.y,
                               z: landmark.z,
                               score: landmark.visibility,
                               name: BLAZEPOSE_KEYPOINTS[i]
                             }))
      }];
    }
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * It returns a single pose or multiple poses based on the maxPose parameter
   * from the `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       maxPoses: Optional. Max number of poses to estimate.
   *       When maxPoses = 1, a single pose is detected, it is usually much more
   *       efficient than maxPoses > 1. When maxPoses > 1, multiple poses are
   *       detected.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   *       enableSmoothing: Optional. Default to true. Smooth pose landmarks
   *       coordinates and visibility scores to reduce jitter.
   *
   * @param timestamp Optional. In milliseconds. This is useful when image is
   *     a tensor, which doesn't have timestamp info. Or to override timestamp
   *     in a video.
   *
   * @return NamedPoseMap `{poses: Pose[], poses3D: Pose[]}`. `poses` contains
   *     a list of `Pose`, each with 33 keypoints, the x and y are scaled to
   *     image size (non-normalized). `poses3D` contains the same list of
   *     `Pose`, the x, y, z are in meters with the origin at the center between
   *     hips.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig: BlazePoseMediaPipeEstimationConfig,
      timestamp?: number): Promise<NamedPoseMap> {
    if (estimationConfig && estimationConfig.flipHorizontal &&
        (estimationConfig.flipHorizontal !== this.selfieMode)) {
      this.selfieMode = estimationConfig.flipHorizontal;
      this.poseSolution.setOptions({
        selfieMode: this.selfieMode,
      });
    }
    await this.poseSolution.send({image: image as pose.InputImage}, timestamp);
    return {poses: this.poses, poses3D: this.poses3D};
  }

  dispose() {
    this.poseSolution.close();
  }

  reset() {
    this.poseSolution.reset();
  }

  initialize(): Promise<void> {
    return this.poseSolution.initialize();
  }
}

/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the BlazePose loading process. Please find more details of each parameters
 * in the documentation of the `BlazePoseMediaPipeModelConfig` interface.
 */
export async function load(modelConfig: BlazePoseMediaPipeModelConfig):
    Promise<PoseDetector> {
  const config = validateModelConfig(modelConfig);
  const result = new BlazePoseMediaPipeDetector(config);
  await result.initialize();
  return result;
}
