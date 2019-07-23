/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

import {Box} from './box';

const LANDMARKS_COUNT = 468;

export class BlazePipeline {
  private blazeface: any;
  private blazemesh: any;
  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private runsWithoutFaceDetector: number;
  private ROIs: any;
  private maxFaces: number;

  constructor(
      blazeface: any, blazemesh: any, meshWidth: number, meshHeight: number,
      maxContinuousChecks: number) {
    this.blazeface = blazeface;
    this.blazemesh = blazemesh;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.runsWithoutFaceDetector = 0;
    this.ROIs = [];

    this.maxFaces = 1;
  }

  /**
   * Calculates face mesh for specific image (468 points).
   *
   * @param {tf.Tensor!} image - image tensor of shape [1, H, W, 3].
   * @return {tf.Tensor?} tensor of 2d coordinates (1, 468, 2)
   */
  predict(image: tf.Tensor) {
    if (this.needsROIsUpdate()) {
      const box = this.blazeface.getSingleBoundingBox(image);
      if (!box) {
        this.clearROIs();
        return null;
      }
      this.updateROIsFromFaceDetector(box.increaseBox());
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    const box = this.ROIs[0];
    const face =
        box.cutFromAndResize(image, [this.meshHeight, this.meshWidth]).div(255);
    const [coords, flag] = this.blazemesh.predict(face);

    const coords2d = tf.reshape(coords, [-1, 3]).slice([0, 0], [-1, 2]);
    const coords2dScaled =
        tf.mul(
              coords2d,
              tf.div(box.getSize(), [this.meshWidth, this.meshHeight]))
            .add(box.startPoint);

    const landmarksBox = this.calculateLandmarksBoundingBox(coords2dScaled);
    this.updateROIsFromFaceDetector(landmarksBox as {} as tf.Tensor[]);

    return [coords2dScaled, landmarksBox, flag]
  }

  updateROIsFromFaceDetector(box: tf.Tensor[]) {
    this.ROIs = [box];
  }

  clearROIs() {
    for (let ROI in this.ROIs) {
      tf.dispose(ROI);
    }
    this.ROIs = [];
  }

  needsROIsUpdate() {
    const ROIsCount = this.ROIs.length;
    const noROIs = ROIsCount == 0;
    const shouldCheckForMoreFaces = ROIsCount != this.maxFaces &&
        this.runsWithoutFaceDetector >= this.maxContinuousChecks;

    return noROIs || shouldCheckForMoreFaces;
  }

  calculateLandmarksBoundingBox(landmarks: tf.Tensor) {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const boxMinMax = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    const box = new Box(boxMinMax.expandDims(0));
    return box.increaseBox();
  }
}
