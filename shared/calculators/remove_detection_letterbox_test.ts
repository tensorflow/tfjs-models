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

// tslint:disable-next-line: no-imports-from-dist
import {expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';

import {Detection, LocationData} from './interfaces/shape_interfaces';
import {removeDetectionLetterbox} from './remove_detection_letterbox';

const EPS = 1e-5;

function createRelativeLocationData(
    xMin: number, yMin: number, width: number, height: number) {
  return {
    relativeBoundingBox:
        {xMin, yMin, width, height, xMax: xMin + width, yMax: yMin + height}
  };
}

function createDetection(
    labels: string[], labelIds: number[], scores: number[],
    locationData: LocationData) {
  const detection: Detection =
      {label: labels, score: scores, labelId: labelIds, locationData};
  return detection;
}

describe('DetectionLetterboxRemovalCalculator', () => {
  const locationData = createRelativeLocationData(0.25, 0.25, 0.25, 0.25);
  const label = 'detected_object';
  const detections = [createDetection([label], [], [0.3], locationData)];
  const keys = ['xMin', 'yMin', 'width', 'height', 'xMax', 'yMax'] as const ;

  it('padding left right.', async () => {
    const locationData = createRelativeLocationData(0.25, 0.25, 0.25, 0.25);
    const label = 'detected_object';
    const detections = [createDetection([label], [], [0.3], locationData)];

    const padding = {left: 0.2, top: 0, right: 0.3, bottom: 0};

    const outputDetections = removeDetectionLetterbox(detections, padding);

    expect(outputDetections.length).toBe(1);
    const outputDetection = outputDetections[0];

    expect(outputDetection.label.length).toBe(1);
    expect(outputDetection.label[0]).toBe(label);
    expect(outputDetection.labelId.length).toBe(0);
    expect(outputDetection.score.length).toBe(1);
    expect(outputDetection.score[0]).toBe(0.3);

    const expectedLocationData: LocationData = {
      relativeBoundingBox: {
        xMin: 0.1,
        yMin: 0.25,
        width: 0.5,
        height: 0.25,
        xMax: 0.6,
        yMax: 0.5
      }
    };

    for (const key of ['xMin', 'yMin', 'width', 'height', 'xMax', 'yMax'] as
         const ) {
      expectNumbersClose(
          outputDetection.locationData.relativeBoundingBox[key],
          expectedLocationData.relativeBoundingBox[key], EPS);
    }
  });

  it('padding top bottom.', async () => {
    const padding = {left: 0, top: 0.2, right: 0, bottom: 0.3};

    const outputDetections = removeDetectionLetterbox(detections, padding);

    expect(outputDetections.length).toBe(1);
    const outputDetection = outputDetections[0];

    expect(outputDetection.label.length).toBe(1);
    expect(outputDetection.label[0]).toBe(label);
    expect(outputDetection.labelId.length).toBe(0);
    expect(outputDetection.score.length).toBe(1);
    expect(outputDetection.score[0]).toBe(0.3);

    const expectedLocationData: LocationData = {
      relativeBoundingBox: {
        xMin: 0.25,
        yMin: 0.1,
        width: 0.25,
        height: 0.5,
        xMax: 0.5,
        yMax: 0.6
      }
    };

    for (const key of keys) {
      expectNumbersClose(
          outputDetection.locationData.relativeBoundingBox[key],
          expectedLocationData.relativeBoundingBox[key], EPS);
    }
  });
});
