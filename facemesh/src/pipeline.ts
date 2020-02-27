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

import {Box, createBox, cutBoxFromImageAndResize, enlargeBox, getBoxSize, scaleBoxCoordinates} from './box';

export type Prediction = {
  coords: tf.Tensor2D,
  scaledCoords: tf.Tensor2D,
  box: Box,
  flag: tf.Scalar
};

// Container for the coordinates of the facial bounding box returned by
// blazeface.
type BlazeBox = {
  startEndTensor: tf.Tensor2D,
  startPoint: tf.Tensor2D,
  endPoint: tf.Tensor2D
};

const LANDMARKS_COUNT = 468;
const MERGE_REGIONS_OF_INTEREST_IOU_THRESHOLD = 0.25;

// The Pipeline coordinates between the bounding box and skeleton models.
export class Pipeline {
  // MediaPipe model for detecting facial bounding boxes.
  private boundingBoxDetector: blazeface.BlazeFaceModel;
  // MediaPipe model for detecting facial mesh.
  private meshDetector: tfconv.GraphModel;

  // An array of facial bounding boxes.
  private regionsOfInterest: Box[];

  private meshWidth: number;
  private meshHeight: number;
  private maxContinuousChecks: number;
  private maxFaces: number;
  private runsWithoutFaceDetector: number;

  constructor(
      blazeface: blazeface.BlazeFaceModel, meshDetector: tfconv.GraphModel,
      meshWidth: number, meshHeight: number, maxContinuousChecks: number,
      maxFaces: number) {
    this.boundingBoxDetector = blazeface;
    this.meshDetector = meshDetector;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.maxFaces = maxFaces;

    this.runsWithoutFaceDetector = 0;
    this.regionsOfInterest = [];
  }

  /**
   * @param input - tensor of shape [1, H, W, 3].
   */
  async predict(input: tf.Tensor4D): Promise<Prediction[]> {
    if (this.shouldUpdateRegionsOfInterest()) {
      const {boxes, scaleFactor} =
          await this.boundingBoxDetector.getBoundingBoxes(
              input,
              false,  // whether to return tensors
              false   // whether to annotate facial bounding boxes with landmark
                      // information
          );

      if (!boxes.length) {
        this.clearRegionsOfInterest();
        return null;
      }

      const scaledBoxes = boxes.map((prediction: BlazeBox) => {
        const cpuBox = createBox(
            prediction.startPoint.squeeze().arraySync() as [number, number],
            prediction.endPoint.squeeze().arraySync() as [number, number]);

        prediction.startPoint.dispose();
        prediction.endPoint.dispose();
        prediction.startEndTensor.dispose();

        return enlargeBox(
            scaleBoxCoordinates(cpuBox, scaleFactor as [number, number]));
      });

      this.updateRegionsOfInterest(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => this.regionsOfInterest.map((box: Box, i) => {
      const face = cutBoxFromImageAndResize(box, input, [
                     this.meshHeight, this.meshWidth
                   ]).div(255);

      // First returned item is 'contours', which is included in the
      // coordinates.
      const [, flag, coords] =
          this.meshDetector.predict(
              face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

      const coordsReshaped = tf.reshape(coords, [-1, 3]);
      const boxSize = getBoxSize(box);
      const normalizedBoxSize =
          [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight, 1];
      const scaledCoords =
          tf.mul(coordsReshaped, normalizedBoxSize).add([...box.topLeft, 0]);

      const landmarksBox = this.calculateLandmarksBoundingBox(scaledCoords);
      this.regionsOfInterest[i] = landmarksBox;

      return {
        coords: coordsReshaped,
        scaledCoords,
        box: landmarksBox,
        flag: flag.squeeze()
      } as Prediction;
    }));
  }

  // Update regions of interest using intersection-over-union thresholding.
  updateRegionsOfInterest(boxes: Box[]) {
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      const previousBox = this.regionsOfInterest[i];
      let iou = 0;

      if (previousBox && previousBox.topLeft) {
        const [boxStartX, boxStartY] = box.topLeft;
        const [boxEndX, boxEndY] = box.bottomRight;
        const [prevStartX, prevStartY] = previousBox.topLeft;
        const [prevEndX, prevEndY] = previousBox.bottomRight;

        const xStartMax = Math.max(boxStartX, prevStartX);
        const yStartMax = Math.max(boxStartY, prevStartY);
        const xEndMin = Math.min(boxEndX, prevEndX);
        const yEndMin = Math.min(boxEndY, prevEndY);

        const intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
        const boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
        const previousBoxArea =
            (prevEndX - prevStartX) * (prevEndY - boxStartY);
        iou = intersection / (boxArea + previousBoxArea - intersection);
      }

      if (iou > MERGE_REGIONS_OF_INTEREST_IOU_THRESHOLD) {
        this.regionsOfInterest[i] = previousBox;
      } else {
        this.regionsOfInterest[i] = box;
      }
    }

    this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
  }

  clearRegionsOfInterest() {
    this.regionsOfInterest = [];
  }

  shouldUpdateRegionsOfInterest(): boolean {
    const roisCount = this.regionsOfInterest.length;
    const noROIs = roisCount === 0;
    const shouldCheckForMoreFaces = roisCount !== this.maxFaces &&
        this.runsWithoutFaceDetector >= this.maxContinuousChecks;

    return this.maxFaces === 1 ? noROIs : noROIs || shouldCheckForMoreFaces;
  }

  calculateLandmarksBoundingBox(landmarks: tf.Tensor): Box {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const xMin = xs.min().squeeze().arraySync() as number;
    const xMax = xs.max().squeeze().arraySync() as number;
    const yMin = ys.min().squeeze().arraySync() as number;
    const yMax = ys.max().squeeze().arraySync() as number;

    return enlargeBox(createBox([xMin, yMin], [xMax, yMax]));
  }
}
