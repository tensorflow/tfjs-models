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
import * as faceDetection from '@mediapipe/face_detection';
import * as tf from '@tensorflow/tfjs-core';

import {MEDIAPIPE_FACE_DETECTOR_KEYPOINTS} from '../constants';
import {FaceDetector} from '../face_detector';
import {getBoundingBox} from '../shared/calculators/association_norm_rect';
import {Keypoint} from '../shared/calculators/interfaces/common_interfaces';
import {BoundingBox} from '../shared/calculators/interfaces/shape_interfaces';
import {Face, FaceDetectorInput} from '../types';

import {validateModelConfig} from './detector_utils';
import {MediaPipeFaceDetectorMediaPipeEstimationConfig, MediaPipeFaceDetectorMediaPipeModelConfig} from './types';

/**
 * MediaPipe detector class.
 */
export class MediaPipeFaceDetectorMediaPipe implements FaceDetector {
  private readonly faceDetectorSolution: faceDetection.FaceDetection;

  // This will be filled out by asynchronous calls to onResults. They will be
  // stable after `await send` is called on the faces solution.
  private width = 0;
  private height = 0;
  private faces: Face[];

  private selfieMode = false;

  // Should not be called outside.
  constructor(config: MediaPipeFaceDetectorMediaPipeModelConfig) {
    this.faceDetectorSolution = new faceDetection.FaceDetection({
      locateFile: (path, base) => {
        if (config.solutionPath) {
          const solutionPath = config.solutionPath.replace(/\/+$/, '');
          return `${solutionPath}/${path}`;
        }
        return `${base}/${path}`;
      }
    });
    this.faceDetectorSolution.setOptions(
        {selfieMode: this.selfieMode, model: config.modelType});
    this.faceDetectorSolution.onResults(results => {
      this.height = results.image.height;
      this.width = results.image.width;
      this.faces = [];
      if (results.detections !== null) {
        for (const detection of results.detections) {
          this.faces.push(this.normalizedToAbsolute(
              detection.landmarks, getBoundingBox(detection.boundingBox)));
        }
      }
    });
  }

  private normalizedToAbsolute(
      landmarks: faceDetection.NormalizedLandmarkList,
      boundingBox: BoundingBox): Face {
    const keypoints = landmarks.map((landmark, i) => {
      const keypoint: Keypoint = {
        x: landmark.x * this.width,
        y: landmark.y * this.height,
        name: MEDIAPIPE_FACE_DETECTOR_KEYPOINTS[i]
      };

      return keypoint;
    });

    const boundingBoxScaled = {
      xMin: boundingBox.xMin * this.width,
      yMin: boundingBox.yMin * this.height,
      xMax: boundingBox.xMax * this.width,
      yMax: boundingBox.yMax * this.height,
      width: boundingBox.width * this.width,
      height: boundingBox.height * this.height
    };

    return {keypoints, box: boundingBoxScaled};
  }

  /**
   * Estimates faces for an image or video frame.
   *
   * It returns a single face or multiple faces based on the maxFaces
   * parameter passed to the constructor of the class.
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   *       staticImageMode: Optional. Defaults to false. Currently unused in
   * this implementation. Image input types are assumed to be static images, and
   * video inputs are assumed to be non static images.
   *
   * @return An array of `Face`s.
   */
  async estimateFaces(
      input: FaceDetectorInput,
      estimationConfig?: MediaPipeFaceDetectorMediaPipeEstimationConfig):
      Promise<Face[]> {
    if (estimationConfig && estimationConfig.flipHorizontal &&
        (estimationConfig.flipHorizontal !== this.selfieMode)) {
      this.selfieMode = estimationConfig.flipHorizontal;
      this.faceDetectorSolution.setOptions({
        selfieMode: this.selfieMode,
      });
    }
    // Cast to GL TexImageSource types.
    input = input instanceof tf.Tensor ?
        new ImageData(
            await tf.browser.toPixels(input), input.shape[1], input.shape[0]) :
        input;
    await this.faceDetectorSolution.send(
        {image: input as faceDetection.InputImage});
    return this.faces;
  }

  dispose() {
    this.faceDetectorSolution.close();
  }

  reset() {
    this.faceDetectorSolution.reset();
    this.width = 0;
    this.height = 0;
    this.faces = null;
    this.selfieMode = false;
  }

  initialize(): Promise<void> {
    return this.faceDetectorSolution.initialize();
  }
}

/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeFaceDetector loading process. Please find more details of each
 * parameters in the documentation of the
 * `MediaPipeFaceDetectorMediaPipeModelConfig` interface.
 */
export async function load(
    modelConfig: MediaPipeFaceDetectorMediaPipeModelConfig):
    Promise<FaceDetector> {
  const config = validateModelConfig(modelConfig);
  const detector = new MediaPipeFaceDetectorMediaPipe(config);
  await detector.initialize();
  return detector;
}
