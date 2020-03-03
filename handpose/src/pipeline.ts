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
import {rotate as rotateWebgl} from './rotate_gpu';
import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix, rotatePoint} from './util';

const PALM_BOX_SHIFT_VECTOR: [number, number] = [0, -0.4];
const PALM_BOX_ENLARGE_FACTOR = 3;
const PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2];

const HAND_BOX_SHIFT_VECTOR: [number, number] = [0, -0.1];
const HAND_BOX_ENLARGE_FACTOR = 1.65;

// The Pipeline coordinates between the bounding box and skeleton models.
export class HandPipeline {
  // MediaPipe model for detecting hand bounding box.
  private boundingBoxDetector: HandDetector;
  // MediaPipe model for detecting hand mesh.
  private meshDetector: tfconv.GraphModel;

  private maxHandsNumber: number;
  private maxContinuousChecks: number;
  private detectionConfidence: number;
  private meshWidth: number;
  private meshHeight: number;

  // An array of hand bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutHandDetector = 0;

  constructor(
      boundingBoxDetector: HandDetector, meshDetector: tfconv.GraphModel,
      meshWidth: number, meshHeight: number, maxContinuousChecks: number,
      detectionConfidence: number) {
    this.boundingBoxDetector = boundingBoxDetector;
    this.meshDetector = meshDetector;
    this.maxContinuousChecks = maxContinuousChecks;
    this.detectionConfidence = detectionConfidence;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;

    this.maxHandsNumber = 1;  // TODO: Add multi-hand support.
  }

  /**
   * Finds a hand in the input image.
   *
   * @param input - tensor of shape [1, H, W, 3].
   */
  async estimateHand(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                     HTMLImageElement|HTMLCanvasElement) {
    const savedWebglPackDepthwiseConvFlag =
        tf.env().get('WEBGL_PACK_DEPTHWISECONV');
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);

    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return input.toFloat().expandDims(0);
    });

    const useFreshBox = this.shouldUpdateRegionsOfInterest();

    if (useFreshBox === true) {
      const boundingBoxPrediction =
          this.boundingBoxDetector.estimateHandBounds(image);
      if (boundingBoxPrediction === null) {
        this.regionsOfInterest = [];
        return null;
      }

      this.updateRegionsOfInterest(
          boundingBoxPrediction, true /*force update*/);
      this.runsWithoutHandDetector = 0;
    } else {
      this.runsWithoutHandDetector++;
    }

    const scaledCoords = tf.tidy(() => {
      const currentBox = this.regionsOfInterest[0];
      const angle = this.calculateRotation(currentBox);

      const palmCenter = getBoxCenter(currentBox);
      const palmCenterNormalized: [number, number] =
          [palmCenter[0] / image.shape[2], palmCenter[1] / image.shape[1]];
      const rotatedImage: tf.Tensor4D =
          rotateWebgl(image, angle, 0, palmCenterNormalized);
      const rotationMatrix = buildRotationMatrix(-angle, palmCenter);

      let box: Box;
      if (useFreshBox === true) {
        const rotatedPalmLandmarks: Array<[number, number]> =
            currentBox.palmLandmarks.map(
                (coord: [number, number]): [number, number] => {
                  const homogeneousCoordinate =
                      [...coord, 1] as [number, number, number];
                  return rotatePoint(homogeneousCoordinate, rotationMatrix);
                });

        const boxAroundPalm =
            this.calculateLandmarksBoundingBox(rotatedPalmLandmarks);
        // boxAroundPalm only surrounds the palm - therefore we shift it
        // upwards so it will capture fingers once enlarged / squarified.
        const shiftedBox = shiftBox(boxAroundPalm, PALM_BOX_SHIFT_VECTOR);
        box = enlargeBox(squarifyBox(shiftedBox), PALM_BOX_ENLARGE_FACTOR);
      } else {
        box = currentBox;
      }

      const croppedInput = cutBoxFromImageAndResize(
          box, rotatedImage, [this.meshWidth, this.meshHeight]);
      const handImage = croppedInput.div(255);

      const [flag, keypoints] =
          this.meshDetector.predict(handImage) as tf.Tensor[];
      if (flag.squeeze().arraySync() < this.detectionConfidence) {
        this.regionsOfInterest = [];
        return null;
      }

      const rawCoords = tf.reshape(keypoints, [-1, 3]).arraySync() as
          Array<[number, number, number]>;

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

      const coords = coordsRotated.map((coord: [number, number, number]) => {
        return [
          coord[0] + originalBoxCenter[0], coord[1] + originalBoxCenter[1],
          coord[2]
        ];
      });

      // The MediaPipe hand mesh model is trained on hands with empty space
      // around them, so we still need to shift / enlarge the box even though
      // the box surrounds the entire hand.
      const boxAroundHand = this.calculateLandmarksBoundingBox(coords);
      const shiftedBoxAroundHand =
          shiftBox(boxAroundHand, HAND_BOX_SHIFT_VECTOR);
      const nextBoundingBox: Box = enlargeBox(
          squarifyBox(shiftedBoxAroundHand), HAND_BOX_ENLARGE_FACTOR);

      const palmLandmarks: Array<[number, number]> = [];
      for (let i = 0; i < PALM_LANDMARK_IDS.length; i++) {
        palmLandmarks.push(coords[PALM_LANDMARK_IDS[i]] as [number, number]);
      }
      nextBoundingBox.palmLandmarks = palmLandmarks;

      this.updateRegionsOfInterest(nextBoundingBox, false /* force replace */);
      return coords;
    });

    image.dispose();

    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    return scaledCoords;
  }

  private calculateLandmarksBoundingBox(landmarks: number[][]) {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);
    const startPoint: [number, number] = [Math.min(...xs), Math.min(...ys)];
    const endPoint: [number, number] = [Math.max(...xs), Math.max(...ys)];
    return {startPoint, endPoint, landmarks};
  }

  private calculateRotation(box: Box) {
    let keypointsArray = box.palmLandmarks;
    return computeRotation(keypointsArray[0], keypointsArray[2]);
  }

  private updateRegionsOfInterest(box: Box, force: boolean) {
    if (force) {
      this.regionsOfInterest = [box];
    } else {
      const prev = this.regionsOfInterest[0];
      let iou = 0;

      if (prev && prev.startPoint) {
        const boxStartEnd = box.startPoint.concat(box.endPoint);
        const prevStartEnd = prev.startPoint.concat(prev.endPoint);

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

      this.regionsOfInterest[0] = iou > 0.8 ? prev : box;
    }
  }

  private shouldUpdateRegionsOfInterest() {
    const roisCount = this.regionsOfInterest.length;

    return roisCount !== this.maxHandsNumber ||
        this.runsWithoutHandDetector >= this.maxContinuousChecks;
  }
}
