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
import * as faceMesh from '@mediapipe/face_mesh';
import * as tf from '@tensorflow/tfjs-core';

import {MEDIAPIPE_FACE_MESH_KEYPOINTS} from '../constants';
import {FaceLandmarksDetector} from '../face_landmarks_detector';
import {Keypoint} from '../shared/calculators/interfaces/common_interfaces';
import {landmarksToDetection} from '../shared/calculators/landmarks_to_detection';
import {Face, FaceLandmarksDetectorInput} from '../types';

import {validateModelConfig} from './detector_utils';
import {MediaPipeFaceMeshMediaPipeEstimationConfig, MediaPipeFaceMeshMediaPipeModelConfig} from './types';

/**
 * MediaPipe detector class.
 */
class MediaPipeFaceMeshMediaPipeLandmarksDetector implements
    FaceLandmarksDetector {
  private readonly faceMeshSolution: faceMesh.FaceMesh;

  // This will be filled out by asynchronous calls to onResults. They will be
  // stable after `await send` is called on the faces solution.
  private width = 0;
  private height = 0;
  private faces: Face[];

  private selfieMode = false;

  // Should not be called outside.
  constructor(config: MediaPipeFaceMeshMediaPipeModelConfig) {
    this.faceMeshSolution = new faceMesh.FaceMesh({
      locateFile: (path, base) => {
        if (config.solutionPath) {
          const solutionPath = config.solutionPath.replace(/\/+$/, '');
          return `${solutionPath}/${path}`;
        }
        return `${base}/${path}`;
      }
    });
    this.faceMeshSolution.setOptions({
      refineLandmarks: config.refineLandmarks,
      selfieMode: this.selfieMode,
      maxNumFaces: config.maxFaces,
    });
    this.faceMeshSolution.onResults((results) => {
      this.height = results.image.height;
      this.width = results.image.width;
      this.faces = [];
      if (results.multiFaceLandmarks !== null) {
        const landmarksList = results.multiFaceLandmarks;

        for (let i = 0; i < landmarksList.length; i++) {
          const keypoints = this.translateOutput(landmarksList[i]);
          this.faces.push({
            keypoints,
            box:
                landmarksToDetection(keypoints).locationData.relativeBoundingBox
          });
        }
      }
    });
  }

  private translateOutput(landmarks: faceMesh.NormalizedLandmarkList) {
    const keypoints = landmarks.map((landmark, i) => {
      const keypoint: Keypoint = {
        x: landmark.x * this.width,
        y: landmark.y * this.height,
        z: landmark.z * this.width,
      };

      const name = MEDIAPIPE_FACE_MESH_KEYPOINTS.get(i);
      if (name != null) {
        keypoint.name = name;
      }

      return keypoint;
    });
    return keypoints;
  }

  /**
   * Estimates faces for an image or video frame.
   *
   * It returns a single face or multiple faces based on the maxFaceMesh
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
      input: FaceLandmarksDetectorInput,
      estimationConfig?: MediaPipeFaceMeshMediaPipeEstimationConfig):
      Promise<Face[]> {
    if (estimationConfig && estimationConfig.flipHorizontal &&
        (estimationConfig.flipHorizontal !== this.selfieMode)) {
      this.selfieMode = estimationConfig.flipHorizontal;
      this.faceMeshSolution.setOptions({
        selfieMode: this.selfieMode,
      });
    }
    // Cast to GL TexImageSource types.
    input = input instanceof tf.Tensor ?
        new ImageData(
            await tf.browser.toPixels(input), input.shape[1], input.shape[0]) :
        input;
    await this.faceMeshSolution.send({image: input as faceMesh.InputImage});
    return this.faces;
  }

  dispose() {
    this.faceMeshSolution.close();
  }

  reset() {
    this.faceMeshSolution.reset();
    this.width = 0;
    this.height = 0;
    this.faces = null;
    this.selfieMode = false;
  }

  initialize(): Promise<void> {
    return this.faceMeshSolution.initialize();
  }
}

/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeFaceMesh loading process. Please find more details of each
 * parameters in the documentation of the
 * `MediaPipeFaceMeshMediaPipeModelConfig` interface.
 */
export async function load(modelConfig: MediaPipeFaceMeshMediaPipeModelConfig):
    Promise<FaceLandmarksDetector> {
  const config = validateModelConfig(modelConfig);
  const detector = new MediaPipeFaceMeshMediaPipeLandmarksDetector(config);
  await detector.initialize();
  return detector;
}
