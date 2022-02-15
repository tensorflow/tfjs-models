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

import {Keypoint} from './interfaces/common_interfaces';
import {LandmarksRefinementConfig} from './interfaces/config_interfaces';

function getNumberOfRefinedLandmarks(refinements: LandmarksRefinementConfig[]) {
  // Gather all used indexes.
  const indices: number[] = [].concat.apply(
      [], refinements.map(refinement => refinement.indexesMapping));

  if (indices.length === 0) {
    throw new Error('There should be at least one landmark in indexes mapping');
  }

  let minIndex = indices[0], maxIndex = indices[0];

  const uniqueIndices = new Set(indices);

  uniqueIndices.forEach(index => {
    minIndex = Math.min(minIndex, index);
    maxIndex = Math.max(maxIndex, index);
  });

  // Check that indxes start with 0 and there is no gaps between min and max
  // indexes.
  const numIndices = uniqueIndices.size;

  if (minIndex !== 0) {
    throw new Error(
        `Indexes are expected to start with 0 instead of ${minIndex}`);
  }

  if (maxIndex + 1 !== numIndices) {
    throw new Error(`Indexes should have no gaps but ${
        maxIndex - numIndices + 1} indexes are missing`);
  }

  return numIndices;
}

function refineXY(
    indexesMapping: number[], landmarks: Keypoint[],
    refinedLandmarks: Keypoint[]) {
  for (let i = 0; i < landmarks.length; ++i) {
    const landmark = landmarks[i];
    const refinedLandmark = {x: landmark.x, y: landmark.y};
    refinedLandmarks[indexesMapping[i]] = refinedLandmark;
  }
}

function getZAverage(landmarks: Keypoint[], indexes: number[]) {
  let zSum = 0;
  for (let i = 0; i < indexes.length; ++i) {
    zSum += landmarks[indexes[i]].z;
  }
  return zSum / indexes.length;
}

function refineZ(
    indexesMapping: number[],
    zRefinement: LandmarksRefinementConfig['zRefinement'],
    landmarks: Keypoint[], refinedLandmarks: Keypoint[]) {
  if (typeof zRefinement === 'string') {
    switch (zRefinement) {
      case 'copy': {
        for (let i = 0; i < landmarks.length; ++i) {
          refinedLandmarks[indexesMapping[i]].z = landmarks[i].z;
        }
        break;
      }
      case 'none':
      default: {
        // Do nothing and keep Z that is already in refined landmarks.
        break;
      }
    }
  } else {
    const zAverage = getZAverage(refinedLandmarks, zRefinement);
    for (let i = 0; i < indexesMapping.length; ++i) {
      refinedLandmarks[indexesMapping[i]].z = zAverage;
    }
  }
}

/**
 * Refine one set of landmarks with another.
 *
 * @param allLandmarks List of landmarks to use for refinement. They will be
 *     applied to the output in the provided order. Each list should be non
 *     empty and contain the same amount of landmarks as indexes in mapping.
 * @param refinements Refinement instructions for input landmarks.
 *
 * @returns A list of refined landmarks.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_refinement_calculator.cc
export function landmarksRefinement(
    allLandmarks: Keypoint[][],
    refinements: LandmarksRefinementConfig[]): Keypoint[] {
  // Initialize refined landmarks list.
  const numRefinedLandmarks = getNumberOfRefinedLandmarks(refinements);
  const refinedLandmarks: Keypoint[] = new Array(numRefinedLandmarks);

  // Apply input landmarks to output refined landmarks in provided order.
  for (let i = 0; i < allLandmarks.length; ++i) {
    const landmarks = allLandmarks[i];
    const refinement = refinements[i];

    if (landmarks.length !== refinement.indexesMapping.length) {
      // Check number of landmarks in mapping and stream are the same.
      throw new Error(`There are ${
          landmarks.length} refinement landmarks while mapping has ${
          refinement.indexesMapping.length}`);
    }

    // Refine X and Y.
    refineXY(refinement.indexesMapping, landmarks, refinedLandmarks);

    // Refine Z.
    refineZ(
        refinement.indexesMapping, refinement.zRefinement, landmarks,
        refinedLandmarks);

    // Visibility and presence are not currently refined and are left as `0`.
  }

  return refinedLandmarks;
}
