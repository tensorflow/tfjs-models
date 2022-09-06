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

import {normalizeRadians} from './image_utils';
import {ImageSize} from './interfaces/common_interfaces';
import {DetectionToRectConfig} from './interfaces/config_interfaces';
import {BoundingBox, Detection, LocationData, Rect} from './interfaces/shape_interfaces';

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc
export function computeRotation(
    detection: Detection, imageSize: ImageSize, config: DetectionToRectConfig) {
  const locationData = detection.locationData;
  const startKeypoint = config.rotationVectorStartKeypointIndex;
  const endKeypoint = config.rotationVectorEndKeypointIndex;

  let targetAngle;

  if (config.rotationVectorTargetAngle) {
    targetAngle = config.rotationVectorTargetAngle;
  } else {
    targetAngle = Math.PI * config.rotationVectorTargetAngleDegree / 180;
  }

  const x0 = locationData.relativeKeypoints[startKeypoint].x * imageSize.width;
  const y0 = locationData.relativeKeypoints[startKeypoint].y * imageSize.height;
  const x1 = locationData.relativeKeypoints[endKeypoint].x * imageSize.width;
  const y1 = locationData.relativeKeypoints[endKeypoint].y * imageSize.height;

  const rotation =
      normalizeRadians(targetAngle - Math.atan2(-(y1 - y0), x1 - x0));

  return rotation;
}

function rectFromBox(box: BoundingBox) {
  return {
    xCenter: box.xMin + box.width / 2,
    yCenter: box.yMin + box.height / 2,
    width: box.width,
    height: box.height,
  };
}

function normRectFromKeypoints(locationData: LocationData) {
  const keypoints = locationData.relativeKeypoints;
  if (keypoints.length <= 1) {
    throw new Error('2 or more keypoints required to calculate a rect.');
  }
  let xMin = Number.MAX_VALUE, yMin = Number.MAX_VALUE, xMax = Number.MIN_VALUE,
      yMax = Number.MIN_VALUE;

  keypoints.forEach(keypoint => {
    xMin = Math.min(xMin, keypoint.x);
    xMax = Math.max(xMax, keypoint.x);
    yMin = Math.min(yMin, keypoint.y);
    yMax = Math.max(yMax, keypoint.y);
  });

  return {
    xCenter: (xMin + xMax) / 2,
    yCenter: (yMin + yMax) / 2,
    width: xMax - xMin,
    height: yMax - yMin
  };
}

function detectionToNormalizedRect(
    detection: Detection, conversionMode: 'boundingbox'|'keypoints') {
  const locationData = detection.locationData;
  return conversionMode === 'boundingbox' ?
      rectFromBox(locationData.relativeBoundingBox) :
      normRectFromKeypoints(locationData);
}

function detectionToRect(
    detection: Detection,
    conversionMode: 'boundingbox'|'keypoints',
    imageSize?: ImageSize,
    ): Rect {
  const locationData = detection.locationData;

  let rect: Rect;
  if (conversionMode === 'boundingbox') {
    rect = rectFromBox(locationData.boundingBox);
  } else {
    rect = normRectFromKeypoints(locationData);
    const {width, height} = imageSize;

    rect.xCenter = Math.round(rect.xCenter * width);
    rect.yCenter = Math.round(rect.yCenter * height);
    rect.width = Math.round(rect.width * width);
    rect.height = Math.round(rect.height * height);
  }

  return rect;
}

// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc
export function calculateDetectionsToRects(
    detection: Detection, conversionMode: 'boundingbox'|'keypoints',
    outputType: 'rect'|'normRect', imageSize?: ImageSize,
    rotationConfig?: DetectionToRectConfig): Rect {
  const rect: Rect = outputType === 'rect' ?
      detectionToRect(detection, conversionMode, imageSize) :
      detectionToNormalizedRect(detection, conversionMode);

  if (rotationConfig) {
    rect.rotation = computeRotation(detection, imageSize, rotationConfig);
  }

  return rect;
}
