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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box, cutBoxFromImageAndResize, enlargeBox, getBoxCenter, getBoxSize, shiftBox, squarifyBox} from './box';
import {HandDetector} from './hand';
import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix, rotatePoint, TransformationMatrix} from './util';

const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.8;

const PALM_BOX_SHIFT_VECTOR: [number, number] = [0, -0.4];
const PALM_BOX_ENLARGE_FACTOR = 3;

const HAND_BOX_SHIFT_VECTOR: [number, number] = [0, -0.1];
const HAND_BOX_ENLARGE_FACTOR = 1.65;

const PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2];
const PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0;
const PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2;

export type Coords3D = Array<[number, number, number]>;
type Coords2D = Array<[number, number]>;

export interface Prediction {
  handInViewConfidence: number;
  landmarks: Coords3D;
  boundingBox: {topLeft: [number, number], bottomRight: [number, number]};
}

// The Pipeline coordinates between the bounding box and skeleton models.
export class HandPipeline {
  private readonly maxHandsNumber: number;

  // An array of hand bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutHandDetector = 0;

  constructor(
      private readonly boundingBoxDetector: HandDetector
      /* MediaPipe model for detecting hand bounding box */,
      private readonly meshDetector: tfconv.GraphModel
      /* MediaPipe model for detecting hand mesh */,
      private readonly meshWidth: number, private readonly meshHeight: number,
      private readonly maxContinuousChecks: number,
      private readonly detectionConfidence: number) {
    this.maxHandsNumber = 1;  // TODO(annxingyuan): Add multi-hand support.
  }

  // Get the bounding box surrounding the hand, given palm landmarks.
  private getBoxForPalmLandmarks(
      palmLandmarks: Coords2D, rotationMatrix: TransformationMatrix): Box {
    const rotatedPalmLandmarks: Coords2D =
        palmLandmarks.map((coord: [number, number]): [number, number] => {
          const homogeneousCoordinate =
              [...coord, 1] as [number, number, number];
          return rotatePoint(homogeneousCoordinate, rotationMatrix);
        });

    const boxAroundPalm =
        this.calculateLandmarksBoundingBox(rotatedPalmLandmarks);
    // boxAroundPalm only surrounds the palm - therefore we shift it
    // upwards so it will capture fingers once enlarged + squarified.
    return enlargeBox(
        squarifyBox(shiftBox(boxAroundPalm, PALM_BOX_SHIFT_VECTOR)),
        PALM_BOX_ENLARGE_FACTOR);
  }

  // Get the bounding box surrounding the hand, given all hand landmarks.
  private getBoxForHandLandmarks(landmarks: Coords3D): Box {
    // The MediaPipe hand mesh model is trained on hands with empty space
    // around them, so we still need to shift / enlarge boxAroundHand even
    // though it surrounds the entire hand.
    const boundingBox = this.calculateLandmarksBoundingBox(landmarks);
    const boxAroundHand: Box = enlargeBox(
        squarifyBox(shiftBox(boundingBox, HAND_BOX_SHIFT_VECTOR)),
        HAND_BOX_ENLARGE_FACTOR);

    const palmLandmarks: Coords2D = [];
    for (let i = 0; i < PALM_LANDMARK_IDS.length; i++) {
      palmLandmarks.push(
          landmarks[PALM_LANDMARK_IDS[i]].slice(0, 2) as [number, number]);
    }
    boxAroundHand.palmLandmarks = palmLandmarks;

    return boxAroundHand;
  }

  // Scale, rotate, and translate raw keypoints from the model so they map to
  // the input coordinates.
  private transformRawCoords(
      rawCoords: Coords3D, box: Box, angle: number,
      rotationMatrix: TransformationMatrix): Coords3D {
    const boxSize = getBoxSize(box);
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
    const boxCenter = [...getBoxCenter(box), 1];

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

  async estimateHand(image: tf.Tensor4D): Promise<Prediction> {
    const useFreshBox = this.shouldUpdateRegionsOfInterest();
    if (useFreshBox === true) {
      const boundingBoxPrediction =
          await this.boundingBoxDetector.estimateHandBounds(image);
      if (boundingBoxPrediction === null) {
        image.dispose();
        this.regionsOfInterest = [];
        return null;
      }

      this.updateRegionsOfInterest(
          boundingBoxPrediction, true /*force update*/);
      this.runsWithoutHandDetector = 0;
    } else {
      this.runsWithoutHandDetector++;
    }

    // Rotate input so the hand is vertically oriented.
    const currentBox = this.regionsOfInterest[0];
    const angle = computeRotation(
        currentBox.palmLandmarks[PALM_LANDMARKS_INDEX_OF_PALM_BASE],
        currentBox.palmLandmarks[PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]);

    const palmCenter = getBoxCenter(currentBox);
    const palmCenterNormalized: [number, number] =
        [palmCenter[0] / image.shape[2], palmCenter[1] / image.shape[1]];
    const rotatedImage =
        tf.image.rotateWithOffset(image, angle, 0, palmCenterNormalized);

    const rotationMatrix = buildRotationMatrix(-angle, palmCenter);

    let box: Box;
    // The bounding box detector only detects palms, so if we're using a fresh
    // bounding box prediction, we have to construct the hand bounding box from
    // the palm keypoints.
    if (useFreshBox === true) {
      box =
          this.getBoxForPalmLandmarks(currentBox.palmLandmarks, rotationMatrix);
    } else {
      box = currentBox;
    }

    const croppedInput = cutBoxFromImageAndResize(
        box, rotatedImage, [this.meshWidth, this.meshHeight]);
    const handImage = tf.div(croppedInput, 255);
    croppedInput.dispose();
    rotatedImage.dispose();

    let prediction;
    if (tf.getBackend() === 'webgl') {
      // Currently tfjs-core does not pack depthwiseConv because it fails for
      // very large inputs (https://github.com/tensorflow/tfjs/issues/1652).
      // TODO(annxingyuan): call tf.enablePackedDepthwiseConv when available
      // (https://github.com/tensorflow/tfjs/issues/2821)
      const savedWebglPackDepthwiseConvFlag =
          tf.env().get('WEBGL_PACK_DEPTHWISECONV');
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
      prediction =
          this.meshDetector.predict(handImage) as [tf.Tensor, tf.Tensor];
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    } else {
      prediction =
          this.meshDetector.predict(handImage) as [tf.Tensor, tf.Tensor];
    }

    const [flag, keypoints] = prediction;

    handImage.dispose();

    const flagValue = flag.dataSync()[0];
    flag.dispose();

    if (flagValue < this.detectionConfidence) {
      keypoints.dispose();
      this.regionsOfInterest = [];
      return null;
    }

    const keypointsReshaped = tf.reshape(keypoints, [-1, 3]);
    // Calling arraySync() because the tensor is very small so it's not worth
    // calling await array().
    const rawCoords = keypointsReshaped.arraySync() as Coords3D;
    keypoints.dispose();
    keypointsReshaped.dispose();

    const coords =
        this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
    const nextBoundingBox = this.getBoxForHandLandmarks(coords);

    this.updateRegionsOfInterest(nextBoundingBox, false /* force replace */);

    const result: Prediction = {
      landmarks: coords,
      handInViewConfidence: flagValue,
      boundingBox: {
        topLeft: nextBoundingBox.startPoint,
        bottomRight: nextBoundingBox.endPoint
      }
    };

    return result;
  }

  private calculateLandmarksBoundingBox(landmarks: number[][]): Box {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);
    const startPoint: [number, number] = [Math.min(...xs), Math.min(...ys)];
    const endPoint: [number, number] = [Math.max(...xs), Math.max(...ys)];
    return {startPoint, endPoint};
  }

  // Updates regions of interest if the intersection over union between
  // the incoming and previous regions falls below a threshold.
  private updateRegionsOfInterest(box: Box, forceUpdate: boolean): void {
    if (forceUpdate) {
      this.regionsOfInterest = [box];
    } else {
      const previousBox = this.regionsOfInterest[0];
      let iou = 0;

      if (previousBox != null && previousBox.startPoint != null) {
        const [boxStartX, boxStartY] = box.startPoint;
        const [boxEndX, boxEndY] = box.endPoint;
        const [previousBoxStartX, previousBoxStartY] = previousBox.startPoint;
        const [previousBoxEndX, previousBoxEndY] = previousBox.endPoint;

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

      this.regionsOfInterest[0] =
          iou > UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD ? previousBox : box;
    }
  }

  private shouldUpdateRegionsOfInterest(): boolean {
    const roisCount = this.regionsOfInterest.length;

    return roisCount !== this.maxHandsNumber ||
        this.runsWithoutHandDetector >= this.maxContinuousChecks;
  }
}
