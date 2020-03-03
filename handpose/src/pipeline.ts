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

import {Box, cutBoxFromImageAndResize, enlargeBox, getBoxCenter, getBoxSize} from './box';
import {HandDetector} from './hand';
import {rotate as rotateWebgl} from './rotate_gpu';
import {buildRotationMatrix, computeRotation, dot, invertTransformMatrix} from './util';

const FRESH_BOX_SHIFT_VECTOR = [0, -0.4];
const FRESH_BOX_ENLARGE_FACTOR = 3;

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
      return (input as tf.Tensor).toFloat().expandDims(0);
    });

    const useFreshBox = this.shouldUpdateRegionsOfInterest();

    if (useFreshBox === true) {
      const box = this.boundingBoxDetector.estimateHandBounds(image);
      if (box === null) {
        this.regionsOfInterest = [];
        return null;
      }

      this.updateRegionsOfInterest(box, true /*force update*/);
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
        const rotatedLandmarks: Array<[number, number]> =
            currentBox.landmarks.map(
                (coord: [number, number]): [number, number] => {
                  const homogeneousCoordinate = [...coord, 1];
                  return [
                    dot(homogeneousCoordinate, rotationMatrix[0]),
                    dot(homogeneousCoordinate, rotationMatrix[1])
                  ];
                });

        const boxAroundRotatedLandmarks =
            this.calculateLandmarksBoundingBox(rotatedLandmarks);
        // boxAroundRotatedLandmarks only surrounds palm - so, we shift it
        // upwards so it will capture fingers once enlarged and squarified.
        const shiftedBox =
            this.shiftBox(boxAroundRotatedLandmarks, FRESH_BOX_SHIFT_VECTOR);
        box = enlargeBox(
            this.makeSquareBox(shiftedBox), FRESH_BOX_ENLARGE_FACTOR);
      } else {
        box = currentBox;
      }

      const croppedInput = cutBoxFromImageAndResize(
          box, rotatedImage, [this.meshWidth, this.meshHeight]);
      const handImage = croppedInput.div(255);

      const prediction = this.meshDetector.predict(handImage) as tf.Tensor[];

      const output_keypoints = prediction[prediction.length - 1];
      const coords = tf.reshape(output_keypoints, [-1, 3]).arraySync() as
          Array<[number, number, number]>;

      const boxSize = getBoxSize(box);
      const scaleFactor =
          [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];

      const coordsScaled = coords.map((coord: [number, number, number]) => {
        return [
          scaleFactor[0] * (coord[0] - 128), scaleFactor[1] * (coord[1] - 128),
          coord[2]
        ];
      });

      const coords_rotation_matrix = buildRotationMatrix(angle, [0, 0]);
      const coordsRotated =
          coordsScaled.map((coord: [number, number, number]) => {
            return [
              dot(coord, coords_rotation_matrix[0]),
              dot(coord, coords_rotation_matrix[1])
            ];
          });

      const inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
      const numerator = [...getBoxCenter(box), 1];

      const original_center = [
        dot(numerator, inverseRotationMatrix[0]),
        dot(numerator, inverseRotationMatrix[1]),
        dot(numerator, inverseRotationMatrix[2])
      ];

      const coordsResult =
          coordsRotated.map((coord: [number, number, number]) => {
            return [
              coord[0] + original_center[0], coord[1] + original_center[1],
              coord[2] + original_center[2]
            ];
          });

      const landmarks_ids = [0, 5, 9, 13, 17, 1, 2];

      const selected_landmarks = [];
      for (let i = 0; i < landmarks_ids.length; i++) {
        selected_landmarks.push(coordsResult[landmarks_ids[i]]);
      }

      const landmarks_box = this.calculateLandmarksBoundingBox(
          coordsResult as [number, number][]);

      const landmarks_box_shifted = this.shiftBox(landmarks_box, [0, -0.1]);
      const landmarks_box_shifted_squarified =
          this.makeSquareBox(landmarks_box_shifted);

      const nextBoundingBox: Box =
          enlargeBox(landmarks_box_shifted_squarified, 1.65);
      nextBoundingBox.landmarks = selected_landmarks as [number, number][];

      this.updateRegionsOfInterest(nextBoundingBox, false /* force replace */);

      const handFlag = prediction[0].squeeze().arraySync() as number;
      if (handFlag < this.detectionConfidence) {
        this.regionsOfInterest = [];
        return null;
      }

      return coordsResult;
    });

    image.dispose();

    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    return scaledCoords;
  }

  makeSquareBox(box: Box) {
    const centers = getBoxCenter(box);
    const size = getBoxSize(box);
    const maxEdge = Math.max(...size);

    const halfSize = maxEdge / 2;
    const startPoint: [number, number] =
        [centers[0] - halfSize, centers[1] - halfSize];
    const endPoint: [number, number] =
        [centers[0] + halfSize, centers[1] + halfSize];

    return {startPoint, endPoint, landmarks: box.landmarks};
  }

  shiftBox(box: Box, shifts: number[]) {
    const boxSize = [
      box.endPoint[0] - box.startPoint[0], box.endPoint[1] - box.startPoint[1]
    ];
    const absoluteShifts = [boxSize[0] * shifts[0], boxSize[1] * shifts[1]];
    const startPoint: [number, number] = [
      box.startPoint[0] + absoluteShifts[0],
      box.startPoint[1] + absoluteShifts[1]
    ];
    const endPoint: [number, number] = [
      box.endPoint[0] + absoluteShifts[0], box.endPoint[1] + absoluteShifts[1]
    ];
    return {startPoint, endPoint, landmarks: box.landmarks};
  }

  calculateLandmarksBoundingBox(landmarks: Array<[number, number]>) {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);
    const startPoint: [number, number] = [Math.min(...xs), Math.min(...ys)];
    const endPoint: [number, number] = [Math.max(...xs), Math.max(...ys)];
    return {startPoint, endPoint, landmarks};
  }

  calculateRotation(box: Box) {
    let keypointsArray = box.landmarks;
    return computeRotation(keypointsArray[0], keypointsArray[2]);
  }

  updateRegionsOfInterest(box: Box, force: boolean) {
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

  shouldUpdateRegionsOfInterest() {
    const rois_count = this.regionsOfInterest.length;

    return rois_count !== this.maxHandsNumber ||
        this.runsWithoutHandDetector >= this.maxContinuousChecks;
  }
}
