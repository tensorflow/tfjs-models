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

import {Box, createBox, cutBoxFromImageAndResize, disposeBox, enlargeBox, getBoxSize, scaleBoxCoordinates} from './box';

export type Prediction = {
  coords: tf.Tensor2D|tf.Tensor3D,
  scaledCoords: tf.Tensor2D|tf.Tensor3D,
  box: Box,
  flag: tf.Scalar
};

const LANDMARKS_COUNT = 468;

export class Pipeline {
  private blazeface: blazeface.BlazeFaceModel;
  private mesh: tfconv.GraphModel;
  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private runsWithoutFaceDetector: number;
  private rois: Box[];
  private maxFaces: number;

  constructor(
      blazeface: blazeface.BlazeFaceModel, mesh: tfconv.GraphModel,
      meshWidth: number, meshHeight: number, maxContinuousChecks: number,
      maxFaces: number) {
    this.blazeface = blazeface;
    this.mesh = mesh;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.runsWithoutFaceDetector = 0;
    this.rois = [];
    this.maxFaces = maxFaces;
  }

  /**
   * @param image - image tensor of shape [1, H, W, 3].
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
              (prediction: Box): Box => enlargeBox(scaleBoxCoordinates(
                  prediction, scaleFactor as [number, number]))));

      boxes.forEach(disposeBox);

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

      const [, flag, coords] =
          this.mesh.predict(face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

      const coordsReshaped = tf.reshape(coords, [-1, 3]);
      const normalizedBox =
          tf.div(getBoxSize(box), [this.meshWidth, this.meshHeight]);
      const scaledCoords =
          tf.mul(
                coordsReshaped,
                normalizedBox.concat(tf.tensor2d([1], [1, 1]), 1))
              .add(box.startPoint.concat(tf.tensor2d([0], [1, 1]), 1));

      const landmarksBox = this.calculateLandmarksBoundingBox(scaledCoords);
      const prev = this.rois[i];
      if (prev) {
        disposeBox(prev);
      }
      this.rois[i] = landmarksBox;

      return {
        coords: coordsReshaped,
        scaledCoords,
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

        const boxStartX = boxStartEnd[0];
        const prevStartX = prevStartEnd[0];
        const boxStartY = boxStartEnd[1];
        const prevStartY = prevStartEnd[1];
        const boxEndX = boxStartEnd[2];
        const prevEndX = prevStartEnd[2];
        const boxEndY = boxStartEnd[3];
        const prevEndY = prevStartEnd[3];

        const xStartMax = Math.max(boxStartX, prevStartX);
        const yStartMax = Math.max(boxStartY, prevStartY);
        const xEndMin = Math.min(boxEndX, prevEndX);
        const yEndMin = Math.min(boxEndY, prevEndY);

        const interArea = (xEndMin - xStartMax) * (yEndMin - yStartMax);

        const boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
        const prevArea = (prevEndX - prevStartX) * (prevEndY - boxStartY);
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
