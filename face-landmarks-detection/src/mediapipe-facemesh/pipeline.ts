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
import {buildRotationMatrix, computeRotation, Coord2D, Coord3D, Coords3D, dot, IDENTITY_MATRIX, invertTransformMatrix, rotatePoint, TransformationMatrix} from './util';

export type Prediction = {
  coords: tf.Tensor2D,        // coordinates of facial landmarks.
  scaledCoords: tf.Tensor2D,  // coordinates normalized to the mesh size.
  box: Box,                   // bounding box of coordinates.
  flag: tf.Scalar             // confidence in presence of a face.
};

const LANDMARKS_COUNT = 468;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;

const MESH_MOUTH_INDEX = 13;
const MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [MESH_MOUTH_INDEX, MESH_ANNOTATIONS['midwayBetweenEyes'][0]];

const BLAZEFACE_MOUTH_INDEX = 3;
const BLAZEFACE_NOSE_INDEX = 2;
const BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [BLAZEFACE_MOUTH_INDEX, BLAZEFACE_NOSE_INDEX];

const LEFT_EYE_OUTLINE = MESH_ANNOTATIONS['leftEyeLower0'];
const LEFT_EYE_BOUNDS =
    [LEFT_EYE_OUTLINE[0], LEFT_EYE_OUTLINE[LEFT_EYE_OUTLINE.length - 1]];
const RIGHT_EYE_OUTLINE = MESH_ANNOTATIONS['rightEyeLower0'];
const RIGHT_EYE_BOUNDS =
    [RIGHT_EYE_OUTLINE[0], RIGHT_EYE_OUTLINE[RIGHT_EYE_OUTLINE.length - 1]];

const IRIS_UPPER_CENTER_INDEX = 3;
const IRIS_LOWER_CENTER_INDEX = 4;
const IRIS_IRIS_INDEX = 71;
const IRIS_NUM_COORDINATES = 76;

// Factor by which to enlarge the box around the eye landmarks so the input
// region matches the expectations of the iris model.
const ENLARGE_EYE_RATIO = 2.3;
const IRIS_MODEL_INPUT_SIZE = 64;

// A mapping from facemesh model keypoints to iris model keypoints.
const MESH_TO_IRIS_INDICES_MAP = [
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

// Replace the raw coordinates returned by facemesh with refined iris model
// coordinates.
// Update the z coordinate to be an average of the original and the new. This
// produces the best visual effect.
function replaceRawCoordinates(
    rawCoords: Coords3D, newCoords: Coords3D, prefix: string, keys?: string[]) {
  for (let i = 0; i < MESH_TO_IRIS_INDICES_MAP.length; i++) {
    const {key, indices} = MESH_TO_IRIS_INDICES_MAP[i];
    const originalIndices = MESH_ANNOTATIONS[`${prefix}${key}`];

    const shouldReplaceAllKeys = keys == null;
    if (shouldReplaceAllKeys || keys.includes(key)) {
      for (let j = 0; j < indices.length; j++) {
        const index = indices[j];

        rawCoords[originalIndices[j]] = [
          newCoords[index][0], newCoords[index][1],
          (newCoords[index][2] + rawCoords[originalIndices[j]][2]) / 2
        ];
      }
    }
  }
}

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

  public irisModel: tfconv.GraphModel|null;

  // An array of facial bounding boxes.
  private regionsOfInterest: Box[] = [];
  private runsWithoutFaceDetector = 0;

  constructor(
      boundingBoxDetector: blazeface.BlazeFaceModel,
      meshDetector: tfconv.GraphModel, meshWidth: number, meshHeight: number,
      maxContinuousChecks: number, maxFaces: number,
      irisModel: tfconv.GraphModel|null) {
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
    const coordsScaled = rawCoords.map(
        coord => ([
          scaleFactor[0] * (coord[0] - this.meshWidth / 2),
          scaleFactor[1] * (coord[1] - this.meshHeight / 2), coord[2]
        ]));

    const coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
    const coordsRotated = coordsScaled.map(
        (coord: Coord3D) =>
            ([...rotatePoint(coord, coordsRotationMatrix), coord[2]]));

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

  private getLeftToRightEyeDepthDifference(rawCoords: Coords3D): number {
    const leftEyeZ = rawCoords[LEFT_EYE_BOUNDS[0]][2];
    const rightEyeZ = rawCoords[RIGHT_EYE_BOUNDS[0]][2];
    return leftEyeZ - rightEyeZ;
  }

  // Returns a box describing a cropped region around the eye fit for passing to
  // the iris model.
  getEyeBox(
      rawCoords: Coords3D, face: tf.Tensor4D, eyeInnerCornerIndex: number,
      eyeOuterCornerIndex: number,
      flip = false): {box: Box, boxSize: [number, number], crop: tf.Tensor4D} {
    const box = squarifyBox(enlargeBox(
        this.calculateLandmarksBoundingBox(
            [rawCoords[eyeInnerCornerIndex], rawCoords[eyeOuterCornerIndex]]),
        ENLARGE_EYE_RATIO));
    const boxSize = getBoxSize(box);
    let crop = tf.image.cropAndResize(
        face, [[
          box.startPoint[1] / this.meshHeight,
          box.startPoint[0] / this.meshWidth, box.endPoint[1] / this.meshHeight,
          box.endPoint[0] / this.meshWidth
        ]],
        [0], [IRIS_MODEL_INPUT_SIZE, IRIS_MODEL_INPUT_SIZE]);
    if (flip) {
      crop = tf.image.flipLeftRight(crop);
    }

    return {box, boxSize, crop};
  }

  // Given a cropped image of an eye, returns the coordinates of the contours
  // surrounding the eye and the iris.
  getEyeCoords(
      eyeData: Float32Array, eyeBox: Box, eyeBoxSize: [number, number],
      flip = false): {rawCoords: Coords3D, iris: Coords3D} {
    const eyeRawCoords: Coords3D = [];
    for (let i = 0; i < IRIS_NUM_COORDINATES; i++) {
      const x = eyeData[i * 3];
      const y = eyeData[i * 3 + 1];
      const z = eyeData[i * 3 + 2];
      eyeRawCoords.push([
        (flip ? (1 - (x / IRIS_MODEL_INPUT_SIZE)) :
                (x / IRIS_MODEL_INPUT_SIZE)) *
                eyeBoxSize[0] +
            eyeBox.startPoint[0],
        (y / IRIS_MODEL_INPUT_SIZE) * eyeBoxSize[1] + eyeBox.startPoint[1], z
      ]);
    }

    return {rawCoords: eyeRawCoords, iris: eyeRawCoords.slice(IRIS_IRIS_INDEX)};
  }

  // The z-coordinates returned for the iris are unreliable, so we take the z
  // values from the surrounding keypoints.
  private getAdjustedIrisCoords(
      rawCoords: Coords3D, irisCoords: Coords3D,
      direction: 'left'|'right'): Coords3D {
    const upperCenterZ =
        rawCoords[MESH_ANNOTATIONS[`${direction}EyeUpper0`]
                                  [IRIS_UPPER_CENTER_INDEX]][2];
    const lowerCenterZ =
        rawCoords[MESH_ANNOTATIONS[`${direction}EyeLower0`]
                                  [IRIS_LOWER_CENTER_INDEX]][2];
    const averageZ = (upperCenterZ + lowerCenterZ) / 2;

    // Iris indices:
    // 0: center | 1: right | 2: above | 3: left | 4: below
    return irisCoords.map((coord: Coord3D, i): Coord3D => {
      let z = averageZ;
      if (i === 2) {
        z = upperCenterZ;
      } else if (i === 4) {
        z = lowerCenterZ;
      }
      return [coord[0], coord[1], z];
    });
  }

  /**
   * Returns an array of predictions for each face in the input.
   * @param input - tensor of shape [1, H, W, 3].
   * @param predictIrises - Whether to return keypoints for the irises.
   */
  async predict(input: tf.Tensor4D, predictIrises: boolean):
      Promise<Prediction[]> {
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
              startPoint: tf.squeeze(prediction.box.startPoint).arraySync() as
                  Coord2D,
              endPoint: tf.squeeze(prediction.box.endPoint).arraySync() as Coord2D
            };

            const scaledBox =
                scaleBoxCoordinates(predictionBoxCPU, scaleFactor as Coord2D);
            const enlargedBox = enlargeBox(scaledBox);
            const squarifiedBox = squarifyBox(enlargedBox);
            return {
              ...squarifiedBox,
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
        let angle = 0;
        // The facial bounding box landmarks could come either from blazeface
        // (if we are using a fresh box), or from the mesh model (if we are
        // reusing an old box).
        const boxLandmarksFromMeshModel =
            box.landmarks.length >= LANDMARKS_COUNT;
        let [indexOfMouth, indexOfForehead] =
            MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;

        if (boxLandmarksFromMeshModel === false) {
          [indexOfMouth, indexOfForehead] =
              BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
        }

        angle = computeRotation(
            box.landmarks[indexOfMouth], box.landmarks[indexOfForehead]);

        const faceCenter =
            getBoxCenter({startPoint: box.startPoint, endPoint: box.endPoint});
        const faceCenterNormalized: Coord2D =
            [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];

        let rotatedImage = input;
        let rotationMatrix = IDENTITY_MATRIX;
        if (angle !== 0) {
          rotatedImage =
              tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);
          rotationMatrix = buildRotationMatrix(-angle, faceCenter);
        }

        const boxCPU = {startPoint: box.startPoint, endPoint: box.endPoint};
        const face: tf.Tensor4D =
            tf.div(cutBoxFromImageAndResize(boxCPU, rotatedImage, [
              this.meshHeight, this.meshWidth
            ]), 255);

        // The first returned tensor represents facial contours, which are
        // included in the coordinates.
        const [, flag, coords] =
            this.meshDetector.predict(
                face) as [tf.Tensor, tf.Tensor2D, tf.Tensor2D];

        const coordsReshaped: tf.Tensor2D = tf.reshape(coords, [-1, 3]);
        let rawCoords = coordsReshaped.arraySync() as Coords3D;

        if (predictIrises) {
          const {box: leftEyeBox, boxSize: leftEyeBoxSize, crop: leftEyeCrop} =
              this.getEyeBox(
                  rawCoords, face, LEFT_EYE_BOUNDS[0], LEFT_EYE_BOUNDS[1],
                  true);
          const {
            box: rightEyeBox,
            boxSize: rightEyeBoxSize,
            crop: rightEyeCrop
          } =
              this.getEyeBox(
                  rawCoords, face, RIGHT_EYE_BOUNDS[0], RIGHT_EYE_BOUNDS[1]);

          const eyePredictions =
              (this.irisModel.predict(
                  tf.concat([leftEyeCrop, rightEyeCrop]))) as tf.Tensor4D;
          const eyePredictionsData = eyePredictions.dataSync() as Float32Array;

          const leftEyeData =
              eyePredictionsData.slice(0, IRIS_NUM_COORDINATES * 3);
          const {rawCoords: leftEyeRawCoords, iris: leftIrisRawCoords} =
              this.getEyeCoords(leftEyeData, leftEyeBox, leftEyeBoxSize, true);

          const rightEyeData =
              eyePredictionsData.slice(IRIS_NUM_COORDINATES * 3);
          const {rawCoords: rightEyeRawCoords, iris: rightIrisRawCoords} =
              this.getEyeCoords(rightEyeData, rightEyeBox, rightEyeBoxSize);

          const leftToRightEyeDepthDifference =
              this.getLeftToRightEyeDepthDifference(rawCoords);
          if (Math.abs(leftToRightEyeDepthDifference) <
              30) {  // User is looking straight ahead.
            replaceRawCoordinates(rawCoords, leftEyeRawCoords, 'left');
            replaceRawCoordinates(rawCoords, rightEyeRawCoords, 'right');
          } else if (leftToRightEyeDepthDifference < 1) {  // User is looking
                                                           // towards the
                                                           // right.
            // If the user is looking to the left or to the right, the iris
            // coordinates tend to diverge too much from the mesh coordinates
            // for them to be merged. So we only update a single contour line
            // above and below the eye.
            replaceRawCoordinates(
                rawCoords, leftEyeRawCoords, 'left',
                ['EyeUpper0', 'EyeLower0']);
          } else {  // User is looking towards the left.
            replaceRawCoordinates(
                rawCoords, rightEyeRawCoords, 'right',
                ['EyeUpper0', 'EyeLower0']);
          }

          const adjustedLeftIrisCoords =
              this.getAdjustedIrisCoords(rawCoords, leftIrisRawCoords, 'left');
          const adjustedRightIrisCoords = this.getAdjustedIrisCoords(
              rawCoords, rightIrisRawCoords, 'right');
          rawCoords = rawCoords.concat(adjustedLeftIrisCoords)
                          .concat(adjustedRightIrisCoords);
        }

        const transformedCoordsData =
            this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
        const transformedCoords = tf.tensor2d(transformedCoordsData);

        const landmarksBox = enlargeBox(
            this.calculateLandmarksBoundingBox(transformedCoordsData));
        const squarifiedLandmarksBox = squarifyBox(landmarksBox);
        this.regionsOfInterest[i] = {
          ...squarifiedLandmarksBox,
          landmarks: transformedCoords.arraySync() as Coords3D
        };

        const prediction: Prediction = {
          coords: tf.tensor2d(rawCoords, [rawCoords.length, 3]),
          scaledCoords: transformedCoords,
          box: landmarksBox,
          flag: tf.squeeze(flag)
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
