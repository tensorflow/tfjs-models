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
import {buildRotationMatrix, computeRotation, Coord2D, Coord3D, Coords3D, dot, IDENTITY_MATRIX, invertTransformMatrix, rotatePoint, TransformationMatrix, xyDistanceBetweenPoints} from './util';

export type Prediction = {
  coords: tf.Tensor2D,        // coordinates of facial landmarks.
  scaledCoords: tf.Tensor2D,  // coordinates normalized to the mesh size.
  box: Box,                   // bounding box of coordinates.
  flag: tf.Scalar             // confidence in presence of a face.
};

const LANDMARKS_COUNT = 468;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;

const MESH_MOUTH_INDEX = 13;
const MESH_LEFT_EYE_INDEX = 386;
const MESH_RIGHT_EYE_INDEX = 159;
const MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [MESH_ANNOTATIONS['noseTip'][0], MESH_ANNOTATIONS['midwayBetweenEyes'][0]];

const BLAZEFACE_MOUTH_INDEX = 3;
const BLAZEFACE_LEFT_EYE_INDEX = 0;
const BLAZEFACE_RIGHT_EYE_INDEX = 1;
const BLAZEFACE_NOSE_INDEX = 2;
const BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES =
    [BLAZEFACE_MOUTH_INDEX, BLAZEFACE_NOSE_INDEX];

const LEFT_EYE_OUTLINE = MESH_ANNOTATIONS['leftEyeLower0'];
const LEFT_EYE_BOUNDS =
    [LEFT_EYE_OUTLINE[0], LEFT_EYE_OUTLINE[LEFT_EYE_OUTLINE.length - 1]];
const RIGHT_EYE_OUTLINE = MESH_ANNOTATIONS['rightEyeLower0'];
const RIGHT_EYE_BOUNDS =
    [RIGHT_EYE_OUTLINE[0], RIGHT_EYE_OUTLINE[RIGHT_EYE_OUTLINE.length - 1]];

const IRIS_UPPER_CENTER_INDEX = 12;
const IRIS_LOWER_CENTER_INDEX = 4;
const IRIS_IRIS_INDEX = 71;

// Factor by which to enlarge the box around the eye landmarks so the input
// region matches the expectations of the iris model.
const ENLARGE_EYE_RATIO = 2.3;
const IRIS_MODEL_INPUT_SIZE = 64;

// Threshold for determining when the face is sufficiently in profile that it
// shouldn't be rotated.
const X_TO_Y_ROTATION_THRESHOLD = 0.3;

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
function replaceRawCoordinates(
    rawCoords: Coords3D, newCoords: Coords3D, prefix: string, keys?: string[]) {
  for (let i = 0; i < MESH_TO_IRIS_INDICES_MAP.length; i++) {
    const {key, indices} = MESH_TO_IRIS_INDICES_MAP[i];
    const originalIndices = MESH_ANNOTATIONS[`${prefix}${key}`];

    if (keys == null || keys.includes(key)) {
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

  // Returns the ratio of the projected XY distance between the eyes, and
  // between the forehead and mouth.
  private getRatioHorizontalToVertical(
      leftEye: Coord3D, rightEye: Coord3D, mouth: Coord3D): number {
    const midwayBetweenEyes: Coord3D =
        [(leftEye[0] + rightEye[0]) / 2, (leftEye[1] + rightEye[1]) / 2, 0];
    const distanceBetweenEyes = xyDistanceBetweenPoints(leftEye, rightEye);
    const distanceBetweenVertical =
        xyDistanceBetweenPoints(midwayBetweenEyes, mouth);
    return distanceBetweenEyes / distanceBetweenVertical;
  }

  // Returns the ratio of the projected left eye bounding box size, and the
  // right eye bounding box size.
  private getRatioLeftToRightEye(leftEyeSize: [number, number], rightEyeSize: [
    number, number
  ]): number {
    const leftTotalSize = leftEyeSize[0] * leftEyeSize[1];
    const rightTotalSize = rightEyeSize[0] * rightEyeSize[1];
    return leftTotalSize / rightTotalSize;
  }

  // Given a cropped image of an eye, returns the coordinates of the contours
  // surrounding the eye and the iris.
  getEyeCoords(
      face: tf.Tensor4D, eyeBox: Box, eyeBoxSize: [number, number],
      flip = false) {
    let eye = tf.image.cropAndResize(
        face, [[
          eyeBox.startPoint[1] / this.meshHeight,
          eyeBox.startPoint[0] / this.meshWidth,
          eyeBox.endPoint[1] / this.meshHeight,
          eyeBox.endPoint[0] / this.meshWidth
        ]],
        [0], [IRIS_MODEL_INPUT_SIZE, IRIS_MODEL_INPUT_SIZE]);

    if (flip === true) {
      eye = tf.image.flipLeftRight(eye);
    }

    const eyePrediction = (this.irisModel.predict(eye) as tf.Tensor).squeeze();
    const eyeRawCoords: Coords3D =
        (eyePrediction.reshape([-1, 3]).arraySync() as Coords3D)
            .map((coord: Coord3D) => ([
                   (flip ? (1 - (coord[0] / IRIS_MODEL_INPUT_SIZE)) :
                           (coord[0] / IRIS_MODEL_INPUT_SIZE)) *
                           eyeBoxSize[0] +
                       eyeBox.startPoint[0],
                   (coord[1] / IRIS_MODEL_INPUT_SIZE) * eyeBoxSize[1] +
                       eyeBox.startPoint[1],
                   coord[2]
                 ]));

    // The z-coordinates returned for the iris are unreliable, so we take the z
    // values from the surrounding keypoints.
    const eyeUpperCenterZ = eyeRawCoords[IRIS_UPPER_CENTER_INDEX][2];
    const eyeLowerCenterZ = eyeRawCoords[IRIS_LOWER_CENTER_INDEX][2];
    const averageZ = (eyeUpperCenterZ + eyeLowerCenterZ) / 2;

    // Iris indices:
    // 0: center | 1: right | 2: above | 3: left | 4: below
    const irisRawCoords =
        eyeRawCoords.slice(IRIS_IRIS_INDEX).map((coord: Coord3D, i) => {
          let z = averageZ;
          if (i === 2) {
            z = eyeUpperCenterZ;
          } else if (i === 4) {
            z = eyeLowerCenterZ;
          }
          return [coord[0], coord[1], z];
        }) as Coords3D;
    return {rawCoords: eyeRawCoords, iris: irisRawCoords};
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
        let angle = 0;
        // The facial bounding box landmarks could come either from blazeface
        // (if we are using a fresh box), or from the mesh model (if we are
        // reusing an old box).
        const boxLandmarksFromMeshModel =
            box.landmarks.length >= LANDMARKS_COUNT;
        let leftEyeIndex = MESH_LEFT_EYE_INDEX;
        let rightEyeIndex = MESH_RIGHT_EYE_INDEX;
        let mouthIndex = MESH_MOUTH_INDEX;
        let [indexOfNose, indexOfForehead] =
            MESH_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;

        if (boxLandmarksFromMeshModel === false) {
          leftEyeIndex = BLAZEFACE_LEFT_EYE_INDEX;
          rightEyeIndex = BLAZEFACE_RIGHT_EYE_INDEX;
          mouthIndex = BLAZEFACE_MOUTH_INDEX;
          [indexOfNose, indexOfForehead] =
              BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES;
        }

        const ratioHorizontalToVertical = this.getRatioHorizontalToVertical(
            box.landmarks[leftEyeIndex], box.landmarks[rightEyeIndex],
            box.landmarks[mouthIndex]);
        // If the face is not in profile...
        if (ratioHorizontalToVertical < X_TO_Y_ROTATION_THRESHOLD) {
          angle = computeRotation(
              box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
        }

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

        if (predictIrises) {
          const leftEyeBox = squarifyBox(enlargeBox(
              this.calculateLandmarksBoundingBox([
                rawCoords[LEFT_EYE_BOUNDS[0]], rawCoords[LEFT_EYE_BOUNDS[1]]
              ]),
              ENLARGE_EYE_RATIO));
          const leftEyeBoxSize = getBoxSize(leftEyeBox);

          const rightEyeBox = squarifyBox(enlargeBox(
              this.calculateLandmarksBoundingBox([
                rawCoords[RIGHT_EYE_BOUNDS[0]], rawCoords[RIGHT_EYE_BOUNDS[1]]
              ]),
              ENLARGE_EYE_RATIO));
          const rightEyeBoxSize = getBoxSize(rightEyeBox);

          const ratioLeftToRightEye =
              this.getRatioLeftToRightEye(leftEyeBoxSize, rightEyeBoxSize);

          // If the user is looking straight ahead...
          if (0.7 < ratioLeftToRightEye && ratioLeftToRightEye < 1.3) {
            const leftEye = this.getEyeCoords(
                face as tf.Tensor4D, leftEyeBox, leftEyeBoxSize, true);
            const leftEyeRawCoords = leftEye.rawCoords;
            const leftIrisRawCoords = leftEye.iris;

            const rightEye = this.getEyeCoords(
                face as tf.Tensor4D, rightEyeBox, rightEyeBoxSize);
            const rightEyeRawCoords = rightEye.rawCoords;
            const rightIrisRawCoords = rightEye.iris;

            rawCoords =
                rawCoords.concat(leftIrisRawCoords).concat(rightIrisRawCoords);
            replaceRawCoordinates(rawCoords, leftEyeRawCoords, 'left');
            replaceRawCoordinates(rawCoords, rightEyeRawCoords, 'right');
          } else if (ratioLeftToRightEye > 1) {  // User is looking towards the
                                                 // right.
            const leftEye = this.getEyeCoords(
                face as tf.Tensor4D, leftEyeBox, leftEyeBoxSize, true);
            const leftEyeRawCoords = leftEye.rawCoords;
            const leftIrisRawCoords = leftEye.iris;

            rawCoords = rawCoords.concat(leftIrisRawCoords);
            replaceRawCoordinates(
                rawCoords, leftEyeRawCoords, 'left',
                ['EyeUpper0', 'EyeLower0']);
          } else {  // User is looking towards the left.
            const rightEye = this.getEyeCoords(
                face as tf.Tensor4D, rightEyeBox, rightEyeBoxSize);
            const rightEyeRawCoords = rightEye.rawCoords;
            const rightIrisRawCoords = rightEye.iris;

            rawCoords = rawCoords.concat(rightIrisRawCoords);
            replaceRawCoordinates(
                rawCoords, rightEyeRawCoords, 'right',
                ['EyeUpper0', 'EyeLower0']);
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
