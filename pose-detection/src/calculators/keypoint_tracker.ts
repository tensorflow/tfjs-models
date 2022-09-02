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

import {Keypoint} from '../shared/calculators/interfaces/common_interfaces';

import {Pose} from '../types';

import {Track} from './interfaces/common_interfaces';
import {TrackerConfig} from './interfaces/config_interfaces';
import {Tracker} from './tracker';

/**
 * KeypointTracker, which tracks poses based on keypoint similarity. This
 * tracker assumes that keypoints are provided in normalized image
 * coordinates.
 */
export class KeypointTracker extends Tracker {
  private readonly keypointThreshold: number;
  private readonly keypointFalloff: number[];
  private readonly minNumKeyoints: number;

  constructor(config: TrackerConfig) {
    super(config);
    this.keypointThreshold =
        config.keypointTrackerParams.keypointConfidenceThreshold;
    this.keypointFalloff = config.keypointTrackerParams.keypointFalloff;
    this.minNumKeyoints = config.keypointTrackerParams.minNumberOfKeypoints;
  }

  /**
   * Computes similarity based on Object Keypoint Similarity (OKS). It's
   * assumed that the keypoints within each `Pose` are in normalized image
   * coordinates. See `Tracker` for more details.
   */
  computeSimilarity(poses: Pose[]): number[][] {
    if (poses.length === 0 || this.tracks.length === 0) {
      return [[]];
    }

    const simMatrix = [];
    for (const pose of poses) {
      const row = [];
      for (const track of this.tracks) {
        row.push(this.oks(pose, track));
      }
      simMatrix.push(row);
    }
    return simMatrix;
  }

  /**
   * Computes the Object Keypoint Similarity (OKS) between a pose and track.
   * This is similar in spirit to the calculation used by COCO keypoint eval:
   * https://cocodataset.org/#keypoints-eval
   * In this case, OKS is calculated as:
   * (1/sum_i d(c_i, c_ti)) * \sum_i exp(-d_i^2/(2*a_ti*x_i^2))*d(c_i, c_ti)
   * where
   *   d(x, y) is an indicator function which only produces 1 if x and y
   *     exceed a given threshold (i.e. keypointThreshold), otherwise 0.
   *   c_i is the confidence of keypoint i from the new pose
   *   c_ti is the confidence of keypoint i from the track
   *   d_i is the Euclidean distance between the pose and track keypoint
   *   a_ti is the area of the track object (the box covering the keypoints)
   *   x_i is a constant that controls falloff in a Gaussian distribution,
   *    computed as 2*keypointFalloff[i].
   * @param pose A `Pose`.
   * @param track A `Track`.
   * @returns The OKS score between the pose and the track. This number is
   * between 0 and 1, and larger values indicate more keypoint similarity.
   */
  private oks(pose: Pose, track: Track): number {
    const boxArea = this.area(track.keypoints) + 1e-6;
    let oksTotal = 0;
    let numValidKeypoints = 0;
    for (let i = 0; i < pose.keypoints.length; ++i) {
      const poseKpt = pose.keypoints[i];
      const trackKpt = track.keypoints[i];
      if (poseKpt.score < this.keypointThreshold ||
          trackKpt.score < this.keypointThreshold) {
        continue;
      }
      numValidKeypoints += 1;
      const dSquared = Math.pow(poseKpt.x - trackKpt.x, 2) +
          Math.pow(poseKpt.y - trackKpt.y, 2);
      const x = 2 * this.keypointFalloff[i];
      oksTotal += Math.exp(-1 * dSquared / (2 * boxArea * x ** 2));
    }
    if (numValidKeypoints < this.minNumKeyoints) {
      return 0.0;
    }
    return oksTotal / numValidKeypoints;
  }

  /**
   * Computes the area of a bounding box that tightly covers keypoints.
   * @param Keypoint[] An array of `Keypoint`s.
   * @returns The area of the object.
   */
  private area(keypoints: Keypoint[]): number {
    const validKeypoint =
        keypoints.filter(kpt => kpt.score > this.keypointThreshold);
    const minX = Math.min(1.0, ...validKeypoint.map(kpt => kpt.x));
    const maxX = Math.max(0.0, ...validKeypoint.map(kpt => kpt.x));
    const minY = Math.min(1.0, ...validKeypoint.map(kpt => kpt.y));
    const maxY = Math.max(0.0, ...validKeypoint.map(kpt => kpt.y));
    return (maxX - minX) * (maxY - minY);
  }
}
