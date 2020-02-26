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

import {Box} from './box';
import {CPUBox} from './cpu_box';
import {HandDetector} from './hand';
import {rotate as rotateWebgl} from './rotate_gpu';
import {buildRotationMatrix, computeRotation, invertTransformMatrix} from './util';

const BRANCH_ON_DETECTION = true;  // whether we branch box scaling / shifting
                                   // logic depending on detection type


function boxFromCPUBox(cpuBox: CPUBox) {
  return new Box(
      tf.tensor([cpuBox.startEndTensor]),
      cpuBox.landmarks ? tf.tensor(cpuBox.landmarks) : null);
}

function cpuBoxFromBox(box: Box) {
  return new CPUBox(
      ((box.startEndTensor as any).arraySync() as
       any)[0] as [number, number, number, number],
      box.landmarks ? box.landmarks.arraySync() as [number, number][] : null);
}

export class HandPipeline {
  private handdetect: HandDetector;
  private handtrackModel: tfconv.GraphModel;
  private runsWithoutHandDetector: number;
  private maxHandsNum: number;
  private rois: any[];
  private maxContinuousChecks: number;
  private detectionConfidence: number;

  constructor(
      handdetect: HandDetector, handtrackModel: tfconv.GraphModel,
      maxContinuousChecks: number, detectionConfidence: number) {
    this.handdetect = handdetect;
    this.handtrackModel = handtrackModel;
    this.maxContinuousChecks = maxContinuousChecks;
    this.detectionConfidence = detectionConfidence;
    this.runsWithoutHandDetector = 0;
    this.maxHandsNum = 1;  // TODO: Add multi-hand support.
    this.rois = [];
  }

  calculateHandPalmCenter(box: any) {
    return tf.gather(box.landmarks, [0, 2]).mean(0);
  }

  /**
   * Calculates hand mesh for specific image (21 points).
   *
   * @param {tf.Tensor!} input - image tensor of shape [1, H, W, 3].
   *
   * @return {tf.Tensor?} tensor of 2d coordinates (1, 21, 3)
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

    const useFreshBox = this.needROIUpdate();

    if (useFreshBox) {
      const start = tf.memory().numTensors;
      const box = this.handdetect.getSingleBoundingBox(image);
      console.log(
          'leaked tensors after detection', tf.memory().numTensors - start);

      if (!box) {
        this.clearROIS();
        return null;
      }

      this.updateROIFromFacedetector(box, true);
      this.runsWithoutHandDetector = 0;
    } else {
      this.runsWithoutHandDetector++;
    }

    const scaledCoords = tf.tidy(() => {
      const width = 256., height = 256.;
      const box = this.rois[0];
      const cpuBox = cpuBoxFromBox(box);

      const angle = this.calculateRotation(cpuBox);

      const handpalm_center = cpuBox.getCenter();
      const handpalm_center_relative = [
        handpalm_center[0] / image.shape[2], handpalm_center[1] / image.shape[1]
      ];
      const rotated_image = rotateWebgl(
          image, angle, 0, handpalm_center_relative as [number, number]);
      const rotationMatrix = buildRotationMatrix(-angle, handpalm_center);
      const palm_rotation_matrix = tf.tensor2d(rotationMatrix as any);

      let box_for_cut, bbRotated, bbShifted, bbSquarified;
      if (!BRANCH_ON_DETECTION || useFreshBox) {
        const numLandmarks = cpuBox.landmarks.length;
        const box_landmarks_homo = cpuBox.landmarks.map(coord => [...coord, 1]);
        const rotated_landmarks =
            tf.matMul(box_landmarks_homo, palm_rotation_matrix, false, true)
                .slice([0, 0], [numLandmarks, 2]);

        bbRotated = this.calculateLandmarksBoundingBox(
            rotated_landmarks.arraySync() as [number, number][]);
        const shiftVector: [number, number] = [0, -0.4];
        bbShifted = this.shiftBox(bbRotated, shiftVector);
        bbSquarified = this.makeSquareBox(bbShifted);
        box_for_cut = bbSquarified.increaseBox(3.0);
      } else {
        box_for_cut = cpuBox;
      }

      const cutted_hand = box_for_cut.cutFromAndResize(
          rotated_image as tf.Tensor4D, [width, height]);

      const handImage = cutted_hand.div(255);

      const output = this.handtrackModel.predict(handImage) as tf.Tensor[];

      const output_keypoints = output[output.length - 1];
      const coords = tf.reshape(output_keypoints, [-1, 3]);

      const boxSize = box_for_cut.getSize();
      const denom = [boxSize[0] / width, boxSize[1] / height, 1];
      const coordsScaled = tf.mul(coords.sub(tf.tensor([128, 128, 0])), denom);

      const coords_rotation_matrix =
          tf.tensor2d(buildRotationMatrix(angle, [0, 0]) as any);

      const coordsRotated =
          tf.matMul(coordsScaled, coords_rotation_matrix, false, true);

      const inverseRotationMatrix =
          tf.tensor2d(invertTransformMatrix(rotationMatrix));

      const numerator = [...box_for_cut.getCenter(), 1];
      const original_center = tf.matMul(
          tf.tensor(numerator, [1, 3]), inverseRotationMatrix, false, true);

      const coordsResult = coordsRotated.add(original_center);

      const landmarks_ids = [0, 5, 9, 13, 17, 1, 2];
      const selected_landmarks = tf.gather(coordsResult, landmarks_ids);

      let nextBoundingBox;
      if (BRANCH_ON_DETECTION) {
        const landmarks_box = this.calculateLandmarksBoundingBox(
            coordsResult.arraySync() as [number, number][]);

        const landmarks_box_shifted = this.shiftBox(landmarks_box, [0, -0.1]);
        const landmarks_box_shifted_squarified =
            this.makeSquareBox(landmarks_box_shifted);
        nextBoundingBox =
            boxFromCPUBox(landmarks_box_shifted_squarified).increaseBox(1.65);
        (nextBoundingBox as any).landmarks = tf.keep(selected_landmarks);
      } else {
        nextBoundingBox = this.calculateLandmarksBoundingBox(
            selected_landmarks.arraySync() as [number, number][]);
      }

      this.updateROIFromFacedetector(
          cpuBoxFromBox(nextBoundingBox as any), false /* force replace */);

      const handFlag =
          ((output[0] as tf.Tensor).arraySync() as number[][])[0][0];
      if (handFlag < this.detectionConfidence) {
        this.clearROIS();
        return null;
      }

      let result = [coordsResult];
      if (location.hash === '#debug') {
        result = result.concat([
          angle, cutted_hand, box as any, bbRotated as any, bbShifted as any,
          bbSquarified as any, nextBoundingBox as any
        ]);
      }

      return result;
    });

    image.dispose();

    tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
    return scaledCoords;
  }

  makeSquareBox(box: CPUBox) {
    const centers = box.getCenter();
    const size = box.getSize();
    const maxEdge = Math.max(...size);

    const halfSize = maxEdge / 2;
    const newStarts = [centers[0] - halfSize, centers[1] - halfSize];
    const newEnds = [centers[0] + halfSize, centers[1] + halfSize];

    return new CPUBox(
        newStarts.concat(newEnds) as [number, number, number, number]);
  }

  shiftBox(box: CPUBox, shifts: number[]) {
    const boxSize = [
      box.endPoint[0] - box.startPoint[0], box.endPoint[1] - box.startPoint[1]
    ];
    const absoluteShifts = [boxSize[0] * shifts[0], boxSize[1] * shifts[1]];
    const newStart = [
      box.startPoint[0] + absoluteShifts[0],
      box.startPoint[1] + absoluteShifts[1]
    ];
    const newEnd = [
      box.endPoint[0] + absoluteShifts[0], box.endPoint[1] + absoluteShifts[1]
    ];
    return new CPUBox(
        newStart.concat(newEnd) as [number, number, number, number]);
  }

  calculateLandmarksBoundingBox(landmarks: [number, number][]) {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);
    const startEnd: [number, number, number, number] =
        [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    return new CPUBox(startEnd, landmarks);
  }

  calculateRotation(box: CPUBox) {
    let keypointsArray = box.landmarks as [number, number][];
    return computeRotation(keypointsArray[0], keypointsArray[2]);
  }

  updateROIFromFacedetector(box: CPUBox, force: boolean) {
    if (force) {
      this.rois = [boxFromCPUBox(box)];
    } else {
      const prev = this.rois[0];
      let iou = 0;

      if (prev && prev.startPoint) {
        const boxStartEnd = box.startEndTensor;
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

      if (iou > 0.8) {
        this.rois[0] = prev;
      } else {
        this.rois[0] = boxFromCPUBox(box);
      }
    }
  }

  clearROIS() {
    for (let roi in this.rois) {
      tf.dispose(roi);
    }
    this.rois = [];
  }

  needROIUpdate() {
    const rois_count = this.rois.length;

    return rois_count !== this.maxHandsNum ||
        this.runsWithoutHandDetector >= this.maxContinuousChecks;
  }
}
