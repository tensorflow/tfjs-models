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
import * as tf from '@tensorflow/tfjs-core';
import {BLAZEPOSE_KEYPOINTS} from '../constants';
import {PoseDetector} from '../pose_detector';
import {Mask} from '../shared/calculators/interfaces/common_interfaces';
import {assertMaskValue, toImageDataLossy, toTensorLossy} from '../shared/calculators/mask_util';
import {Pose, PoseDetectorInput} from '../types';

import {validateModelConfig} from './detector_utils';
import {BlazePoseMediaPipeEstimationConfig, BlazePoseMediaPipeModelConfig} from './types';

class BlazePoseMediaPipeMask implements Mask {
  constructor(private mask: pose.GpuBuffer) {}

  async toCanvasImageSource() {
    return this.mask;
  }

  async toImageData() {
    return toImageDataLossy(this.mask);
  }

  async toTensor() {
    return toTensorLossy(this.mask);
  }

  getUnderlyingType() {
    return 'canvasimagesource' as const ;
  }
}

function maskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  return 'person';
}
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
      smoothLandmarks: config.enableSmoothing,
      enableSegmentation: config.enableSegmentation,
      smoothSegmentation: config.smoothSegmentation,
      selfieMode: this.selfieMode,
    });
    this.poseSolution.onResults((results) => {
      this.height = results.image.height;
      this.width = results.image.width;
      if (results.poseLandmarks == null) {
        this.poses = [];
      } else {
        const pose = this.translateOutput(
            results.poseLandmarks, results.poseWorldLandmarks);
        if (results.segmentationMask) {
          pose.segmentation = {
            maskValueToLabel,
            mask: new BlazePoseMediaPipeMask(results.segmentationMask)
          };
        }
        this.poses = [pose];
      }
    });
  }

  private translateOutput(
      pose: pose.NormalizedLandmarkList, pose3D?: pose.LandmarkList): Pose {
    const output: Pose = {
      keypoints: pose.map((landmark, i) => ({
                            x: landmark.x * this.width,
                            y: landmark.y * this.height,
                            z: landmark.z,
                            score: landmark.visibility,
                            name: BLAZEPOSE_KEYPOINTS[i]
                          }))
    };

    if (pose3D != null) {
      output.keypoints3D = pose3D.map((landmark, i) => ({
                                        x: landmark.x,
                                        y: landmark.y,
                                        z: landmark.z,
                                        score: landmark.visibility,
                                        name: BLAZEPOSE_KEYPOINTS[i]
                                      }));
    }

    return output;
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
   *       When maxPoses = 1, a single pose is detected, it is usually much
   *       more efficient than maxPoses > 1. When maxPoses > 1, multiple poses
   *       are detected.
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
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig: BlazePoseMediaPipeEstimationConfig,
      timestamp?: number): Promise<Pose[]> {
    if (estimationConfig && estimationConfig.flipHorizontal &&
        (estimationConfig.flipHorizontal !== this.selfieMode)) {
      this.selfieMode = estimationConfig.flipHorizontal;
      this.poseSolution.setOptions({
        selfieMode: this.selfieMode,
      });
    }
    // Cast to GL TexImageSource types.
    image = image instanceof tf.Tensor ?
        new ImageData(
            await tf.browser.toPixels(image), image.shape[1], image.shape[0]) :
        image;
    await this.poseSolution.send({image: image as pose.InputImage}, timestamp);
    return this.poses;
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
