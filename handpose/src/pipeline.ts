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
import {rotate as rotateCpu} from './rotate_cpu';
import {rotate as rotateWebgl} from './rotate_gpu';
import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix, rotatePoint} from './util';

const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.8;

const PALM_BOX_SHIFT_VECTOR: [number, number] = [0, -0.4];
const PALM_BOX_ENLARGE_FACTOR = 3;

const HAND_BOX_SHIFT_VECTOR: [number, number] = [0, -0.1];
const HAND_BOX_ENLARGE_FACTOR = 1.65;

const PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2];
const PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0;
const PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2;

type Coords3D = Array<[number, number, number]>;
type Coords2D = Array<[number, number]>;

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipHandHorizontal(coords: Coords3D, width: number): Coords3D {
  return coords.map(
      (coord: [number, number, number]): [number, number, number] => {
        return [width - 1 - coord[0], coord[1], coord[2]];
      });
}

// The Pipeline coordinates between the bounding box and skeleton models.
export class HandPose {
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
   * @param input The image to classify. Can be a tensor, DOM element image,
   * video, or canvas.
   * @param flipHorizontal Whether to flip the hand keypoints horizontally.
   * Should be true for videos that are flipped by default (e.g. webcams).
   */
  async estimateHand(
      input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement,
      flipHorizontal = false): Promise<Coords3D> {
    const savedWebglPackDepthwiseConvFlag =
        tf.env().get('WEBGL_PACK_DEPTHWISECONV');
    tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);

    const [, width] = getInputTensorDimensions(input);

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

    const scaledCoords: Coords3D = tf.tidy(() => {
      const currentBox = this.regionsOfInterest[0];
      const angle = computeRotation(
          currentBox.palmLandmarks[PALM_LANDMARKS_INDEX_OF_PALM_BASE],
          currentBox.palmLandmarks[PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]);

      const palmCenter = getBoxCenter(currentBox);
      const palmCenterNormalized: [number, number] =
          [palmCenter[0] / image.shape[2], palmCenter[1] / image.shape[1]];
      const rotatedImage: tf.Tensor4D = tf.getBackend() === 'webgl' ?
          rotateWebgl(image, angle, 0, palmCenterNormalized) :
          rotateCpu(image, angle, 0, palmCenterNormalized);
      const rotationMatrix = buildRotationMatrix(-angle, palmCenter);

      let box: Box;
      if (useFreshBox === true) {
        const rotatedPalmLandmarks: Coords2D = currentBox.palmLandmarks.map(
            (coord: [number, number]): [number, number] => {
              const homogeneousCoordinate =
                  [...coord, 1] as [number, number, number];
              return rotatePoint(homogeneousCoordinate, rotationMatrix);
            });

        const boxAroundPalm =
            this.calculateLandmarksBoundingBox(rotatedPalmLandmarks);
        // boxAroundPalm only surrounds the palm - therefore we shift it
        // upwards so it will capture fingers once enlarged + squarified.
        box = enlargeBox(
            squarifyBox(shiftBox(boxAroundPalm, PALM_BOX_SHIFT_VECTOR)),
            PALM_BOX_ENLARGE_FACTOR);
      } else {
        box = currentBox;
      }

      const croppedInput = cutBoxFromImageAndResize(
          box, rotatedImage, [this.meshWidth, this.meshHeight]);
      const handImage = croppedInput.div(255);

      const [flag, keypoints] =
          this.meshDetector.predict(handImage) as [tf.Tensor, tf.Tensor];

      if (flag.squeeze().arraySync() < this.detectionConfidence) {
        this.regionsOfInterest = [];
        return null;
      }

      const rawCoords = tf.reshape(keypoints, [-1, 3]).arraySync() as Coords3D;

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

      const coords: Coords3D = coordsRotated.map(
          (coord: [number, number, number]): [number, number, number] => {
            return [
              coord[0] + originalBoxCenter[0], coord[1] + originalBoxCenter[1],
              coord[2]
            ];
          });

      // The MediaPipe hand mesh model is trained on hands with empty space
      // around them, so we still need to shift / enlarge boxAroundHand even
      // though it surrounds the entire hand.
      const boxAroundHand = this.calculateLandmarksBoundingBox(coords);
      const nextBoundingBox: Box = enlargeBox(
          squarifyBox(shiftBox(boxAroundHand, HAND_BOX_SHIFT_VECTOR)),
          HAND_BOX_ENLARGE_FACTOR);

      const palmLandmarks: Coords2D = [];
      for (let i = 0; i < PALM_LANDMARK_IDS.length; i++) {
        palmLandmarks.push(
            coords[PALM_LANDMARK_IDS[i]].slice(0, 2) as [number, number]);
      }
      nextBoundingBox.palmLandmarks = palmLandmarks;

      this.updateRegionsOfInterest(nextBoundingBox, false /* force replace */);

      if (flipHorizontal === true) {
        return flipHandHorizontal(coords, width);
      }
      return coords;
    });

    image.dispose();

    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);

    return scaledCoords;
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
    if (forceUpdate === true) {
      this.regionsOfInterest = [box];
    } else {
      const previousBox = this.regionsOfInterest[0];
      let iou = 0;

      if (previousBox != null && previousBox.startPoint) {
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
