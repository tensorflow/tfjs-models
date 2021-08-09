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

import {Pose} from '../types';
import {Tracker} from './tracker';
import {Track} from './interfaces/common_interfaces';
import {TrackerConfig} from './interfaces/config_interfaces';

/**
 * BoundingBoxTracker, which tracks objects based on bounding box similarity.
 */
export class BoundingBoxTracker extends Tracker {
  private readonly iouThreshold: number;

  constructor(config: TrackerConfig) {
    super(config);
    this.iouThreshold = config.boundingBoxTrackerParams.iouThreshold;
  }

  /**
   * Computes similarity based on intersection-over-union (IoU).
   */
  computeSimilarity(poses: Pose[]): number[][] {
    if (poses.length === 0 || this.tracks.length === 0) {
      return [[]];
    }
    const simMatrix = poses.map(pose => {
      return this.tracks.map(track => {
        const iou = this.iou(pose, track);
        return iou >= this.iouThreshold ? iou : 0.0;
      });
    });
    return simMatrix;
  }

  /**
   * Computes the intersection-over-union (IoU) between a pose and a track.
   */
  private iou(pose: Pose, track: Track): number {
    const xMin = Math.max(pose.box.xMin, track.box.xMin);
    const yMin = Math.max(pose.box.yMin, track.box.yMin);
    const xMax = Math.min(pose.box.xMax, track.box.xMax);
    const yMax = Math.min(pose.box.yMax, track.box.yMax);
    if (xMin >= xMax || yMin >= yMax) {
      return 0.0;
    }
    const intersection = (xMax - xMin) * (yMax - yMin);
    const area_pose = pose.box.width * pose.box.height;
    const area_track = track.box.width * track.box.height;
    return intersection / (area_pose + area_track - intersection);
  }
}