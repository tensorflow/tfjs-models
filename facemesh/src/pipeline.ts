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

import * as blazeface from '@tensorflow-models/blazeface';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box, createBox, cutBoxFromImageAndResize, disposeBox, enlargeBox, getBoxSize, scaleBox} from './box';

export type Prediction = {
  coords: tf.Tensor2D,
  scaledCoords: tf.Tensor2D,
  box: Box,
  flag: tf.Scalar
};

const LANDMARKS_COUNT = 468;

export class Pipeline {
  private blazeface: blazeface.BlazeFaceModel;
  private blazemesh: tfconv.GraphModel;
  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private runsWithoutFaceDetector: number;
  private rois: Box[];
  private maxFaces: number;

  constructor(
      blazeface: blazeface.BlazeFaceModel, blazemesh: tfconv.GraphModel,
      meshWidth: number, meshHeight: number, maxContinuousChecks: number,
      maxFaces: number) {
    this.blazeface = blazeface;
    this.blazemesh = blazemesh;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.runsWithoutFaceDetector = 0;
    this.rois = [];
    this.maxFaces = maxFaces;
  }

  /**
   * @param {tf.Tensor!} image - image tensor of shape [1, H, W, 3].
   * @return an array of predictions for each face
   */
  async predict(image: tf.Tensor4D): Promise<Prediction[]> {
    if (this.needsRoisUpdate()) {
      const returnTensors = false;
      const annotateFace = false;
      const {boxes, scaleFactor} = await this.blazeface.getBoundingBoxes(
          image, returnTensors, annotateFace);

      if (!boxes.length) {
        this.clearROIs();
        return null;
      }

      const scaledBoxes = tf.tidy(
          () => boxes.map(
              (prediction: Box): Box => enlargeBox(
                  scaleBox(prediction, scaleFactor as [number, number]))));

      this.updateRoisFromFaceDetector(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => this.rois.map((roi, i) => {
      const box = roi as Box;
      const face = cutBoxFromImageAndResize(box, image, [
                     this.meshHeight, this.meshWidth
                   ]).div(255);
      // TODO: What are contours? (first argument)
      // change to [coords, flag] for ultralite model
      const [, flag, coords] =
          this.blazemesh.predict(face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

      const coords2d =
          tf.reshape(coords, [-1, 3]).slice([0, 0], [-1, 2]) as tf.Tensor2D;
      const coords2dScaled =
          tf.mul(
                coords2d,
                tf.div(getBoxSize(box), [this.meshWidth, this.meshHeight]))
              .add(box.startPoint) as tf.Tensor2D;

      const landmarksBox = this.calculateLandmarksBoundingBox(coords2dScaled);
      const prev = this.rois[i];
      if (prev) {
        disposeBox(prev);
      }
      this.rois[i] = landmarksBox;

      return {
        coords: coords2d,
        scaledCoords: coords2dScaled,
        box: landmarksBox,
        flag: flag.squeeze()
      } as Prediction;
    }));
  }

  updateRoisFromFaceDetector(boxes: Box[]) {
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const prev = this.rois[i];
      let iou = 0;

      if (prev && prev.startPoint) {
        const boxStartEnd = box.startEndTensor.arraySync()[0];
        const prevStartEnd = prev.startEndTensor.arraySync()[0];

        const xBox = Math.max(boxStartEnd[0], prevStartEnd[0]);
        const yBox = Math.max(boxStartEnd[1], prevStartEnd[1]);
        const xPrev = Math.min(boxStartEnd[2], prevStartEnd[2]);
        const yPrev = Math.min(boxStartEnd[3], prevStartEnd[3]);

        const interArea = (xPrev - xBox) * (yPrev - yBox);

        const boxArea = (boxStartEnd[2] - boxStartEnd[0]) *
            (boxStartEnd[3] - boxStartEnd[1]);
        const prevArea = (prevStartEnd[2] - prevStartEnd[0]) *
            (prevStartEnd[3] - boxStartEnd[1]);
        iou = interArea / (boxArea + prevArea - interArea);
      }

      if (iou > 0.25) {
        this.rois[i] = prev;
        disposeBox(box);
      } else {
        this.rois[i] = box;
        if (prev && prev.startPoint) {
          disposeBox(prev);
        }
      }
    }

    for (let i = boxes.length; i < this.rois.length; i++) {
      const roi = this.rois[i];
      if (roi) {
        disposeBox(roi);
      }
    }

    this.rois = this.rois.slice(0, boxes.length);
  }

  clearROIs() {
    this.rois = [];
  }

  needsRoisUpdate(): boolean {
    const roisCount = this.rois.length;
    const noROIs = roisCount === 0;
    const shouldCheckForMoreFaces = roisCount !== this.maxFaces &&
        this.runsWithoutFaceDetector >= this.maxContinuousChecks;

    return this.maxFaces === 1 ? noROIs : noROIs || shouldCheckForMoreFaces;
  }

  calculateLandmarksBoundingBox(landmarks: tf.Tensor): Box {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const boxMinMax = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    const box = createBox(boxMinMax.expandDims(0));
    return enlargeBox(box);
  }
}
