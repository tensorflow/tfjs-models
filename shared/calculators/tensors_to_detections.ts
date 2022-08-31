/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import * as tf from '@tensorflow/tfjs-core';
import {TensorsToDetectionsConfig} from './interfaces/config_interfaces';
import {AnchorTensor, Detection} from './interfaces/shape_interfaces';

/**
 * Convert result Tensors from object detection models into Detection boxes.
 *
 * @param detectionTensors List of Tensors of type Float32. The list of tensors
 *     can have 2 or 3 tensors. First tensor is the predicted raw
 *     boxes/keypoints. The size of the values must be
 *     (num_boxes * num_predicted_values). Second tensor is the score tensor.
 *     The size of the valuse must be (num_boxes * num_classes). It's optional
 *     to pass in a third tensor for anchors (e.g. for SSD models) depend on the
 *     outputs of the detection model. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param anchor A tensor for anchors. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param config
 */
export async function tensorsToDetections(
    detectionTensors: [tf.Tensor1D, tf.Tensor2D], anchor: AnchorTensor,
    config: TensorsToDetectionsConfig): Promise<Detection[]> {
  const rawScoreTensor = detectionTensors[0];
  const rawBoxTensor = detectionTensors[1];

  // Shape [numOfBoxes, 4] or [numOfBoxes, 12].
  const boxes = decodeBoxes(rawBoxTensor, anchor, config);

  // Filter classes by scores.
  const normalizedScore = tf.tidy(() => {
    let normalizedScore = rawScoreTensor;
    if (config.sigmoidScore) {
      if (config.scoreClippingThresh != null) {
        normalizedScore = tf.clipByValue(
            rawScoreTensor, -config.scoreClippingThresh,
            config.scoreClippingThresh);
      }
      normalizedScore = tf.sigmoid(normalizedScore);
      return normalizedScore;
    }

    return normalizedScore;
  });

  const outputDetections =
      await convertToDetections(boxes, normalizedScore, config);

  tf.dispose([boxes, normalizedScore]);

  return outputDetections;
}

export async function convertToDetections(
    detectionBoxes: tf.Tensor2D, detectionScore: tf.Tensor1D,
    config: TensorsToDetectionsConfig): Promise<Detection[]> {
  const outputDetections: Detection[] = [];
  const detectionBoxesData = await detectionBoxes.data() as Float32Array;
  const detectionScoresData = await detectionScore.data() as Float32Array;

  for (let i = 0; i < config.numBoxes; ++i) {
    if (config.minScoreThresh != null &&
        detectionScoresData[i] < config.minScoreThresh) {
      continue;
    }
    const boxOffset = i * config.numCoords;
    const detection = convertToDetection(
        detectionBoxesData[boxOffset + 0] /* boxYMin */,
        detectionBoxesData[boxOffset + 1] /* boxXMin */,
        detectionBoxesData[boxOffset + 2] /* boxYMax */,
        detectionBoxesData[boxOffset + 3] /* boxXMax */, detectionScoresData[i],
        config.flipVertically, i);
    const bbox = detection.locationData.relativeBoundingBox;

    if (bbox.width < 0 || bbox.height < 0) {
      // Decoded detection boxes could have negative values for width/height
      // due to model prediction. Filter out those boxes since some
      // downstream calculators may assume non-negative values.
      continue;
    }
    // Add keypoints.
    if (config.numKeypoints > 0) {
      const locationData = detection.locationData;
      locationData.relativeKeypoints = [];
      const totalIdx = config.numKeypoints * config.numValuesPerKeypoint;
      for (let kpId = 0; kpId < totalIdx; kpId += config.numValuesPerKeypoint) {
        const keypointIndex = boxOffset + config.keypointCoordOffset + kpId;
        const keypoint = {
          x: detectionBoxesData[keypointIndex + 0],
          y: config.flipVertically ? 1 - detectionBoxesData[keypointIndex + 1] :
                                     detectionBoxesData[keypointIndex + 1]
        };
        locationData.relativeKeypoints.push(keypoint);
      }
    }
    outputDetections.push(detection);
  }

  return outputDetections;
}

function convertToDetection(
    boxYMin: number, boxXMin: number, boxYMax: number, boxXMax: number,
    score: number, flipVertically: boolean, i: number): Detection {
  return {
    score: [score],
    ind: i,
    locationData: {
      relativeBoundingBox: {
        xMin: boxXMin,
        yMin: flipVertically ? 1 - boxYMax : boxYMin,
        xMax: boxXMax,
        yMax: flipVertically ? 1 - boxYMin : boxYMax,
        width: boxXMax - boxXMin,
        height: boxYMax - boxYMin
      }
    }
  };
}

//[xCenter, yCenter, w, h, kp1, kp2, kp3, kp4]
//[yMin, xMin, yMax, xMax, kpX, kpY, kpX, kpY]
function decodeBoxes(
    rawBoxes: tf.Tensor2D, anchor: AnchorTensor,
    config: TensorsToDetectionsConfig): tf.Tensor2D {
  return tf.tidy(() => {
    let yCenter;
    let xCenter;
    let h;
    let w;

    if (config.reverseOutputOrder) {
      // Shape [numOfBoxes, 1].
      xCenter = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 0], [-1, 1]));
      yCenter = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 1], [-1, 1]));
      w = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 2], [-1, 1]));
      h = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 3], [-1, 1]));
    } else {
      yCenter = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 0], [-1, 1]));
      xCenter = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 1], [-1, 1]));
      h = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 2], [-1, 1]));
      w = tf.squeeze(
          tf.slice(rawBoxes, [0, config.boxCoordOffset + 3], [-1, 1]));
    }

    xCenter =
        tf.add(tf.mul(tf.div(xCenter, config.xScale), anchor.w), anchor.x);
    yCenter =
        tf.add(tf.mul(tf.div(yCenter, config.yScale), anchor.h), anchor.y);

    if (config.applyExponentialOnBoxSize) {
      h = tf.mul(tf.exp(tf.div(h, config.hScale)), anchor.h);
      w = tf.mul(tf.exp(tf.div(w, config.wScale)), anchor.w);
    } else {
      h = tf.mul(tf.div(h, config.hScale), anchor.h);
      w = tf.mul(tf.div(w, config.wScale), anchor.h);
    }

    const yMin = tf.sub(yCenter, tf.div(h, 2));
    const xMin = tf.sub(xCenter, tf.div(w, 2));
    const yMax = tf.add(yCenter, tf.div(h, 2));
    const xMax = tf.add(xCenter, tf.div(w, 2));

    // Shape [numOfBoxes, 4].
    let boxes = tf.concat(
        [
          tf.reshape(yMin, [config.numBoxes, 1]),
          tf.reshape(xMin, [config.numBoxes, 1]),
          tf.reshape(yMax, [config.numBoxes, 1]),
          tf.reshape(xMax, [config.numBoxes, 1])
        ],
        1);

    if (config.numKeypoints) {
      for (let k = 0; k < config.numKeypoints; ++k) {
        const keypointOffset =
            config.keypointCoordOffset + k * config.numValuesPerKeypoint;
        let keypointX;
        let keypointY;
        if (config.reverseOutputOrder) {
          keypointX =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset], [-1, 1]));
          keypointY =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1]));
        } else {
          keypointY =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset], [-1, 1]));
          keypointX =
              tf.squeeze(tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1]));
        }
        const keypointXNormalized = tf.add(
            tf.mul(tf.div(keypointX, config.xScale), anchor.w), anchor.x);
        const keypointYNormalized = tf.add(
            tf.mul(tf.div(keypointY, config.yScale), anchor.h), anchor.y);
        boxes = tf.concat(
            [
              boxes, tf.reshape(keypointXNormalized, [config.numBoxes, 1]),
              tf.reshape(keypointYNormalized, [config.numBoxes, 1])
            ],
            1);
      }
    }

    // Shape [numOfBoxes, 4] || [numOfBoxes, 12].
    return boxes as tf.Tensor2D;
  });
}
