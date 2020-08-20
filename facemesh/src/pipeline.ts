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

import {Box, cutBoxFromImageAndResize, enlargeBox, getBoxCenter, getBoxSize, scaleBoxCoordinates, squarifyBox} from './box';
import {MESH_ANNOTATIONS} from './keypoints';
import {buildRotationMatrix, computeRotation, Coord2D, Coord3D, Coords3D, dot, invertTransformMatrix, rotatePoint, TransformationMatrix} from './util';

export type Prediction = {
  coords: tf.Tensor2D,        // coordinates of facial landmarks.
  scaledCoords: tf.Tensor2D,  // coordinates normalized to the mesh size.
  box: Box,                   // bounding box of coordinates.
  flag: tf.Scalar             // confidence in presence of a face.
};

const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;
const LANDMARKS_COUNT = 468;
const MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [1, 168];
const BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [3, 2];
const REPLACEMENT_INDICES = [
  {key: 'EyeUpper0', indices: [9, 10, 11, 12, 13, 14, 15]},
  {key: 'EyeUpper1', indices: [25, 26, 27, 28, 29, 30, 31]},
  {key: 'EyeUpper2', indices: [41, 42, 43, 44, 45, 46, 47]},
  {key: 'EyeLower0', indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
  {key: 'EyeLower1', indices: [16, 17, 18, 19, 20, 21, 22, 23, 24]},
  {key: 'EyeLower2', indices: [32, 33, 34, 35, 36, 37, 38, 39, 40]},
  {key: 'EyeLower3', indices: [54, 55, 56, 57, 58, 59, 60, 61, 62]},
  {key: 'EyebrowUpper', indices: [63, 64, 65, 66, 67, 68, 69, 70]},
  {key: 'EyebrowLower', indices: [48, 49, 50, 51, 52, 53]}
];

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

  private irisModel: tfconv.GraphModel;

  // An array of facial bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutFaceDetector = 0;

  constructor(
      boundingBoxDetector: blazeface.BlazeFaceModel,
      meshDetector: tfconv.GraphModel, meshWidth: number, meshHeight: number,
      maxContinuousChecks: number, maxFaces: number,
      irisModel: tfconv.GraphModel) {
    this.boundingBoxDetector = boundingBoxDetector;
    this.meshDetector = meshDetector;
    this.irisModel = irisModel;
    this.meshWidth = meshWidth;
    this.meshHeight = meshHeight;
    this.maxContinuousChecks = maxContinuousChecks;
    this.maxFaces = maxFaces;
  }

  transformRawCoords(
      rawCoords: Coords3D, box: Box, angle: number,
      rotationMatrix: TransformationMatrix) {
    const boxSize =
        getBoxSize({startPoint: box.startPoint, endPoint: box.endPoint});
    const scaleFactor =
        [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];

    const coordsScaled = rawCoords.map(coord => {
      return [
        scaleFactor[0] * (coord[0] - this.meshWidth / 2),
        scaleFactor[1] * (coord[1] - this.meshHeight / 2), coord[2]
      ];
    });

    const coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    const coordsRotated = coordsScaled.map((coord: Coord3D) => {
      const rotated = rotatePoint(coord, coordsRotationMatrix);
      return [...rotated, coord[2]];
    });

    const inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
    const boxCenter = [
      ...getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint}), 1
    ];

    const originalBoxCenter = [
      dot(boxCenter, inverseRotationMatrix[0]),
      dot(boxCenter, inverseRotationMatrix[1])
    ];

    return coordsRotated.map((coord): Coord3D => ([
                               coord[0] + originalBoxCenter[0],
                               coord[1] + originalBoxCenter[1], coord[2]
                             ]));
  }

  private getEyeSizeRatios(leftEyeSize: [number, number], rightEyeSize: [
    number, number
  ]): number {
    const leftTotalSize = leftEyeSize[0] * leftEyeSize[1];
    const rightTotalSize = rightEyeSize[0] * rightEyeSize[1];
    if (leftTotalSize < rightTotalSize) {
      return rightTotalSize / leftTotalSize;
    }
    return leftTotalSize / rightTotalSize;
  }

  getEyeCoords(
      rawCoords: Coords3D, face: tf.Tensor4D, eyeCornerLeft: number,
      eyeCornerRight: number, flip = false) {
    const eyeBox = squarifyBox(enlargeBox(
        this.calculateLandmarksBoundingBox(
            [rawCoords[eyeCornerLeft], rawCoords[eyeCornerRight]]),
        2.3));
    let eye = tf.image.cropAndResize(
        face, [[
          eyeBox.startPoint[1] / this.meshHeight,
          eyeBox.startPoint[0] / this.meshWidth,
          eyeBox.endPoint[1] / this.meshHeight,
          eyeBox.endPoint[0] / this.meshWidth
        ]],
        [0], [64, 64]);  // [1, 64, 64, 3]

    if (flip === true) {
      eye = tf.image.flipLeftRight(eye);
    }

    const eyePrediction = (this.irisModel.predict(eye) as tf.Tensor).squeeze();
    const eyeBoxSize = getBoxSize(eyeBox);
    const eyeRawCoords: Coords3D =
        (eyePrediction.reshape([-1, 3]).arraySync() as Coords3D)
            .map((coord: Coord3D) => {
              return [
                (flip ? (1 - (coord[0] / 64)) : (coord[0] / 64)) *
                        eyeBoxSize[0] +
                    eyeBox.startPoint[0],
                (coord[1] / 64) * eyeBoxSize[1] + eyeBox.startPoint[1], coord[2]
              ];
            });

    const eyeUpperCenterZ = eyeRawCoords[12][2];
    const eyeLowerCenterZ = eyeRawCoords[4][2];
    const averageZ = (eyeUpperCenterZ + eyeLowerCenterZ) / 2;

    // 0: center
    // 1: right
    // 2: above
    // 3: left
    // 4: below
    const irisRawCoords = eyeRawCoords.slice(71).map((coord: Coord3D, i) => {
      let z = averageZ;
      if (i === 2) {
        z = eyeUpperCenterZ;
      } else if (i === 4) {
        z = eyeLowerCenterZ;
      }

      return [coord[0], coord[1], z];
    }) as Coords3D;

    return {rawCoords: eyeRawCoords, iris: irisRawCoords, size: eyeBoxSize};
  }

  /**
   * Returns an array of predictions for each face in the input.
   *
   * @param input - tensor of shape [1, H, W, 3].
   */
  async predict(input: tf.Tensor4D): Promise<Prediction[]> {
    if (this.shouldUpdateRegionsOfInterest()) {
      const returnTensors = false;
      const annotateFace = true;
      const {boxes, scaleFactor} =
          await this.boundingBoxDetector.getBoundingBoxes(
              input, returnTensors, annotateFace);

      if (boxes.length === 0) {
        this.regionsOfInterest = [];
        return null;
      }

      const scaledBoxes =
          boxes.map((prediction: blazeface.BlazeFacePrediction): Box => {
            const predictionBoxCPU = {
              startPoint: prediction.box.startPoint.squeeze().arraySync() as
                  Coord2D,
              endPoint: prediction.box.endPoint.squeeze().arraySync() as Coord2D
            };

            const scaledBox =
                scaleBoxCoordinates(predictionBoxCPU, scaleFactor as Coord2D);
            const enlargedBox = enlargeBox(scaledBox);
            return {
              ...enlargedBox,
              landmarks: prediction.landmarks.arraySync() as Coords3D
            };
          });

      boxes.forEach((box: {
                      startPoint: tf.Tensor2D,
                      startEndTensor: tf.Tensor2D,
                      endPoint: tf.Tensor2D
                    }) => {
        if (box != null && box.startPoint != null) {
          box.startEndTensor.dispose();
          box.startPoint.dispose();
          box.endPoint.dispose();
        }
      });

      this.updateRegionsOfInterest(scaledBoxes);
      this.runsWithoutFaceDetector = 0;
    } else {
      this.runsWithoutFaceDetector++;
    }

    return tf.tidy(() => {
      return this.regionsOfInterest.map((box, i) => {
        let angle: number;
        // The facial bounding box landmarks could come either from blazeface
        // (if we are using a fresh box), or from the mesh model (if we are
        // reusing an old box).
        const boxLandmarksFromMeshModel =
            box.landmarks.length >= LANDMARKS_COUNT;
        if (boxLandmarksFromMeshModel) {
          const [indexOfNose, indexOfForehead] =
              MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
          angle = computeRotation(
              box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
        } else {
          const [indexOfNose, indexOfForehead] =
              BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
          angle = computeRotation(
              box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
        }

        const faceCenter =
            getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint});
        const faceCenterNormalized: Coord2D =
            [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];

        const rotatedImage =
            tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);

        const rotationMatrix = buildRotationMatrix(-angle, faceCenter);
        const boxCPU = {startPoint: box.startPoint, endPoint: box.endPoint};

        const face = cutBoxFromImageAndResize(boxCPU, rotatedImage, [
                       this.meshHeight, this.meshWidth
                     ]).div(255);

        // The first returned tensor represents facial contours, which are
        // included in the coordinates.
        const [, flag, coords] =
            this.meshDetector.predict(
                face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        let rawCoords = coordsReshaped.arraySync() as Coords3D;

        const leftEye =
            this.getEyeCoords(rawCoords, face as tf.Tensor4D, 362, 263, true);
        const leftEyeRawCoords = leftEye.rawCoords;
        const leftIrisRawCoords = leftEye.iris;

        const rightEye =
            this.getEyeCoords(rawCoords, face as tf.Tensor4D, 33, 133);
        const rightEyeRawCoords = rightEye.rawCoords;
        const rightIrisRawCoords = rightEye.iris;

        const eyeSizeRatios =
            this.getEyeSizeRatios(leftEye.size, rightEye.size);

        if (Math.abs(1 - eyeSizeRatios) < 1) {
          rawCoords =
              rawCoords.concat(leftIrisRawCoords).concat(rightIrisRawCoords);

          for (let i = 0; i < REPLACEMENT_INDICES.length; i++) {
            const {key, indices} = REPLACEMENT_INDICES[i];
            const leftIndices = MESH_ANNOTATIONS[`left${key}`];
            const rightIndices = MESH_ANNOTATIONS[`right${key}`];
            for (let j = 0; j < indices.length; j++) {
              const index = indices[j];
              rawCoords[leftIndices[j]] = leftEyeRawCoords[index];
              rawCoords[rightIndices[j]] = rightEyeRawCoords[index];
            }
          }
        }

        const transformedCoordsData =
            this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
        const transformedCoords = tf.tensor2d(transformedCoordsData);

        const landmarksBox = enlargeBox(
            this.calculateLandmarksBoundingBox(transformedCoordsData));
        this.regionsOfInterest[i] = {
          ...landmarksBox,
          landmarks: transformedCoords.arraySync() as Coords3D
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

      if (iou < UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
        this.regionsOfInterest[i] = box;
      }
    }

    this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
  }

  clearRegionOfInterest(index: number) {
    if (this.regionsOfInterest[index] != null) {
      this.regionsOfInterest = [
        ...this.regionsOfInterest.slice(0, index),
        ...this.regionsOfInterest.slice(index + 1)
      ];
    }
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

  calculateLandmarksBoundingBox(landmarks: Coords3D): Box {
    const xs = landmarks.map(d => d[0]);
    const ys = landmarks.map(d => d[1]);

    const startPoint: Coord2D = [Math.min(...xs), Math.min(...ys)];
    const endPoint: Coord2D = [Math.max(...xs), Math.max(...ys)];
    return {startPoint, endPoint};
  }
}
