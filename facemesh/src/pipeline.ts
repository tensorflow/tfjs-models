/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
  coords: tf.Tensor2D,        // coordinates of facial landmarks.
  scaledCoords: tf.Tensor2D,  // coordinates normalized to the mesh size.
  box: Box,                   // bounding box of coordinates.
  flag: tf.Scalar             // confidence in presence of a face.
};

const LANDMARKS_COUNT = 468;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;

// The Pipeline coordinates between the bounding box and skeleton models.
export class Pipeline {
  // MediaPipe model for detecting facial bounding boxes.
  private boundingBoxDetector: blazeface.BlazeFaceModel;
  // MediaPipe model for detecting facial mesh.
  private meshDetector: tfconv.GraphModel;

  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private maxFaces: number;

  // An array of facial bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutFaceDetector = 0;

  constructor(
      boundingBoxDetector: blazeface.BlazeFaceModel,
      meshDetector: tfconv.GraphModel, meshWidth: number, meshHeight: number,
      maxContinuousChecks: number, maxFaces: number) {
    this.boundingBoxDetector = boundingBoxDetector;
    this.meshDetector = meshDetector;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.maxFaces = maxFaces;
  }

  /**
   * Returns an array of predictions for each face in the input.
   *
   * @param input - tensor of shape [1, H, W, 3].
   */
  async predict(input: tf.Tensor4D): Promise<Prediction[]> {
    if (this.shouldUpdateRegionsOfInterest()) {
      const returnTensors = true;
      const annotateFace = false;
      const {boxes, scaleFactor} =
          await this.boundingBoxDetector.getBoundingBoxes(
              input, returnTensors, annotateFace);

      if (boxes.length === 0) {
        (scaleFactor as tf.Tensor1D).dispose();
        this.clearAllRegionsOfInterest();
        return null;
      }

      const scaledBoxes = boxes.map(
          (prediction: Box): Box => enlargeBox(scaleBoxCoordinates(
              prediction, scaleFactor as [number, number])));
      boxes.forEach(disposeBox);

      this.updateRegionsOfInterest(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => {
      return this.regionsOfInterest.map((box, i) => {
        const face = cutBoxFromImageAndResize(box, input, [
                       this.meshHeight, this.meshWidth
                     ]).div(255);

        // The first returned tensor represents facial contours, which are
        // included in the coordinates.
        const [, flag, coords] =
            this.meshDetector.predict(
                face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        const normalizedBox =
            tf.div(getBoxSize(box), [this.meshWidth, this.meshHeight]);
        const scaledCoords: tf.Tensor2D =
            tf.mul(
                  coordsReshaped,
                  normalizedBox.concat(tf.tensor2d([1], [1, 1]), 1))
                .add(box.startPoint.concat(tf.tensor2d([0], [1, 1]), 1));

        const landmarksBox = this.calculateLandmarksBoundingBox(scaledCoords);
        const previousBox = this.regionsOfInterest[i];
        disposeBox(previousBox);
        this.regionsOfInterest[i] = landmarksBox;

        const prediction: Prediction = {
          coords: coordsReshaped,
          scaledCoords,
          box: landmarksBox,
          flag: flag.squeeze()
        };

        return prediction;
      });
    });
  }

  // Updates regions of interest if the intersection over union between
  // the incoming and previous regions falls below a threshold.
  updateRegionsOfInterest(boxes: Box[]) {
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const previousBox = this.regionsOfInterest[i];
      let iou = 0;

      if (previousBox && previousBox.startPoint) {
        // Computing IOU on the CPU for performance.
        // Using arraySync() rather than await array() because the tensors are
        // very small, so it's not worth the overhead to call await array().
        const [boxStartX, boxStartY, boxEndX, boxEndY] =
            box.startEndTensor.arraySync()[0];
        const [previousBoxStartX, previousBoxStartY, previousBoxEndX, previousBoxEndY] =
            previousBox.startEndTensor.arraySync()[0];

        const xStartMax = Math.max(boxStartX, previousBoxStartX);
        const yStartMax = Math.max(boxStartY, previousBoxStartY);
        const xEndMin = Math.min(boxEndX, previousBoxEndX);
        const yEndMin = Math.min(boxEndY, previousBoxEndY);

        const intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
        const boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
        const previousBoxArea = (previousBoxEndX - previousBoxStartX) *
            (previousBoxEndY - boxStartY);
        iou = intersection / (boxArea + previousBoxArea - intersection);
      }

      if (iou > UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
        disposeBox(box);
      } else {
        this.regionsOfInterest[i] = box;
        disposeBox(previousBox);
      }
    }

    for (let i = boxes.length; i < this.regionsOfInterest.length; i++) {
      disposeBox(this.regionsOfInterest[i]);
    }

    this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
  }

  clearRegionOfInterest(index: number) {
    if (this.regionsOfInterest[index] != null) {
      disposeBox(this.regionsOfInterest[index]);

      this.regionsOfInterest = [
        ...this.regionsOfInterest.slice(0, index),
        ...this.regionsOfInterest.slice(index + 1)
      ];
    }
  }

  clearAllRegionsOfInterest() {
    for (let i = 0; i < this.regionsOfInterest.length; i++) {
      disposeBox(this.regionsOfInterest[i]);
    }

    this.regionsOfInterest = [];
  }

  shouldUpdateRegionsOfInterest(): boolean {
    const roisCount = this.regionsOfInterest.length;
    const noROIs = roisCount === 0;

    if (this.maxFaces === 1 || noROIs) {
      return noROIs;
    }

    return roisCount !== this.maxFaces &&
        this.runsWithoutFaceDetector >= this.maxContinuousChecks;
  }

  calculateLandmarksBoundingBox(landmarks: tf.Tensor): Box {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const boxMinMax = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    const box = createBox(boxMinMax.expandDims(0));
    return enlargeBox(box);
  }
}
