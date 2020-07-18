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

import {Box, createBox, cutBoxFromImageAndResize, disposeBox, enlargeBox, getBoxCenter, getBoxSize, scaleBoxCoordinates} from './box';
import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix, radToDegrees, rotatePoint, TransformationMatrix} from './util';

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

  transformRawCoords(
      rawCoords: any, box: Box, angle: number,
      rotationMatrix: TransformationMatrix) {
    const boxSize = getBoxSize(box).arraySync()[0];
    const scaleFactor =
        [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];

    const coordsScaled = rawCoords.map((coord: [number, number, number]) => {
      return [
        scaleFactor[0] * (coord[0] - this.meshWidth / 2),
        scaleFactor[1] * (coord[1] - this.meshHeight / 2), coord[2]
      ];
    });

    const coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    const coordsRotated =
        coordsScaled.map((coord: [number, number, number]) => {
          const rotated = rotatePoint(coord, coordsRotationMatrix);
          return [...rotated, coord[2]];
        });

    const inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
    const boxCenter =
        [...getBoxCenter(box).arraySync()[0], 1] as [number, number, number];

    const originalBoxCenter = [
      dot(boxCenter, inverseRotationMatrix[0]),
      dot(boxCenter, inverseRotationMatrix[1])
    ];

    return coordsRotated.map(
        (coord: [number, number, number]): [number, number, number] => {
          return [
            coord[0] + originalBoxCenter[0], coord[1] + originalBoxCenter[1],
            coord[2]
          ];
        });
  }

  /**
   * Returns an array of predictions for each face in the input.
   *
   * @param input - tensor of shape [1, H, W, 3].
   */
  async predict(input: tf.Tensor4D): Promise<Prediction[]> {
    if (this.shouldUpdateRegionsOfInterest()) {
      const returnTensors = true;
      const annotateFace = true;
      const {boxes, scaleFactor} =
          await this.boundingBoxDetector.getBoundingBoxes(
              input, returnTensors, annotateFace);

      if (boxes.length === 0) {
        (scaleFactor as tf.Tensor1D).dispose();
        this.clearAllRegionsOfInterest();
        return null;
      }

      const scaledBoxes =
          boxes.map((prediction: blazeface.BlazeFacePrediction): Box => {
            return {
              ...enlargeBox(scaleBoxCoordinates(
                  prediction.box, scaleFactor as [number, number])),
              landmarks: prediction.landmarks.arraySync()
            };
          });
      boxes.forEach(disposeBox);

      this.updateRegionsOfInterest(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => {
      return this.regionsOfInterest.map((box, i) => {
        let angle: number;
        if (box.landmarks.length === 468) {
          angle = computeRotation(box.landmarks[1], box.landmarks[168]);
        } else {
          angle = computeRotation(box.landmarks[3], box.landmarks[2]);
        }

        const faceCenterTensor = getBoxCenter(box);
        const faceCenter = faceCenterTensor.arraySync()[0] as [number, number];
        const faceCenterNormalized: [number, number] =
            [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];

        console.log(angle);

        const rotatedImage =
            tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);

        const rotationMatrix = buildRotationMatrix(-angle, faceCenter);

        const face = cutBoxFromImageAndResize(box, rotatedImage, [
                       this.meshHeight, this.meshWidth
                     ]).div(255);

        // The first returned tensor represents facial contours, which are
        // included in the coordinates.
        const [, flag, coords] =
            this.meshDetector.predict(
                face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        const rawCoords = coordsReshaped.arraySync();
        // const normalizedBox =
        //     tf.div(getBoxSize(box), [this.meshWidth, this.meshHeight]);
        // const scaledCoords: tf.Tensor2D = tf.mul(
        //     coordsReshaped, normalizedBox.concat(tf.tensor2d([1], [1, 1]),
        //     1));
        // .add(box.startPoint.concat(tf.tensor2d([0], [1, 1]), 1));

        const transformedCoordsData =
            this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
        const transformedCoords = tf.tensor2d(transformedCoordsData);

        const landmarksBox =
            this.calculateLandmarksBoundingBox(transformedCoords);
        const previousBox = this.regionsOfInterest[i];
        disposeBox(previousBox);
        this.regionsOfInterest[i] = {
          ...landmarksBox,
          landmarks: coordsReshaped.arraySync()
        };

        const prediction: Prediction = {
          coords: coordsReshaped,
          scaledCoords: transformedCoords,
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
