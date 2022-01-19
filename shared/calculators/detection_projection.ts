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

import {Matrix4x4, matrix4x4ToArray} from './calculate_inverse_matrix';
import {Detection} from './interfaces/shape_interfaces';

function project(
    projectionMatrix: number[], [x, y]: [number, number]): [number, number] {
  return [
    x * projectionMatrix[0] + y * projectionMatrix[1] + projectionMatrix[3],
    x * projectionMatrix[4] + y * projectionMatrix[5] + projectionMatrix[7]
  ];
}
/**
 * Projects detections to a different coordinate system using a provided
 * projection matrix.
 *
 * @param detections A list of detections to project using the provided
 *     projection matrix.
 * @param projectionMatrix Maps data from one coordinate system to     another.
 * @returns detections: A list of projected detections
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detection_projection_calculator.cc
export function detectionProjection(
    detections: Detection[] = [], projectionMatrix: Matrix4x4): Detection[] {
  const flatProjectionMatrix = matrix4x4ToArray(projectionMatrix);

  detections.forEach(detection => {
    const locationData = detection.locationData;

    // Project keypoints.
    locationData.relativeKeypoints.forEach(keypoint => {
      const [x, y] = project(flatProjectionMatrix, [keypoint.x, keypoint.y]);
      keypoint.x = x;
      keypoint.y = y;
    });

    // Project bounding box.
    const box = locationData.relativeBoundingBox;

    let xMin = Number.MAX_VALUE, yMin = Number.MAX_VALUE,
        xMax = Number.MIN_VALUE, yMax = Number.MIN_VALUE;

    [[box.xMin, box.yMin], [box.xMin + box.width, box.yMin],
     [box.xMin + box.width, box.yMin + box.height],
     [box.xMin, box.yMin + box.height]]
        .forEach(coordinate => {
          // a) Define and project box points.
          const [x, y] =
              project(flatProjectionMatrix, coordinate as [number, number]);
          // b) Find new left top and right bottom points for a box which
          // encompases
          //    non-projected (rotated) box.
          xMin = Math.min(xMin, x);
          xMax = Math.max(xMax, x);
          yMin = Math.min(yMin, y);
          yMax = Math.max(yMax, y);
        });
    locationData.relativeBoundingBox =
        {xMin, xMax, yMin, yMax, width: xMax - xMin, height: yMax - yMin};
  });

  return detections;
}
