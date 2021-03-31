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

import * as tfconv from '@tensorflow/tfjs-converter';
//import * as tf from '@tensorflow/tfjs-core';

import {BasePoseDetector, PoseDetector} from '../pose_detector';
//import {InputResolution, Pose, PoseDetectorInput} from '../types';
import {Pose, PoseDetectorInput} from '../types';

import {MOVENET_CONFIG, SINGLE_PERSON_ESTIMATION_CONFIG} from './constants';
import {validateEstimationConfig, validateModelConfig} from './detector_utils';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

/**
 * MoveNet detector class.
 */
export class MoveNetDetector extends BasePoseDetector {
  private maxPoses: number;

  // Should not be called outside.
  private constructor(
      private readonly moveNetModel: tfconv.GraphModel,
      config: MoveNetModelConfig) {
    super();
    console.log('calling super done');
  }

  /**
   * Loads the MoveNet model instance from a checkpoint. The model to be loaded
   * is configurable using the config dictionary ModelConfig. Please find more
   * details in the documentation of the ModelConfig.
   *
   * @param config ModelConfig dictionary that contains parameters for
   * the MoveNet loading process. Please find more details of each parameters
   * in the documentation of the ModelConfig interface.
   */
  static async load(modelConfig: MoveNetModelConfig = MOVENET_CONFIG):
      Promise<PoseDetector> {
    console.log('Constructing MoveNet');
    const config = validateModelConfig(modelConfig);
    console.log('Config:' + config);
//    const defaultUrl = "";
//    const model = await tfconv.loadGraphModel(config.modelUrl || defaultUrl);
    const model: tfconv.GraphModel = new tfconv.GraphModel('http://localhost:8080/movenet/model.json');

    console.log('Returning ' + model);

    return this.constructor(model, config);
  }

  /**
   * Estimates poses for an image or video frame.
   *
   * This does standard ImageNet pre-processing before inferring through the
   * model. The image should pixels should have values [0-255]. It returns a
   * single pose or multiple poses based on the maxPose parameter from the
   * `config`.
   *
   * @param image
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config
   *       maxPoses: Optional. Max number of poses to estimate.
   *       Only 1 pose is supported.
   *
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of `Pose`s.
   */
  async estimatePoses(
      image: PoseDetectorInput,
      estimationConfig:
          MoveNetEstimationConfig = SINGLE_PERSON_ESTIMATION_CONFIG):
      Promise<Pose[]> {
    const config = validateEstimationConfig(estimationConfig);

    if (image == null) {
      return [];
    }

    this.maxPoses = config.maxPoses;

    if (this.maxPoses > 1) {
      throw new Error('Multi-person poses is not implemented yet.');
    }

    const poses: Pose[] = [];

    return poses;
  }

  dispose() {
    this.moveNetModel.dispose();
  }
}
