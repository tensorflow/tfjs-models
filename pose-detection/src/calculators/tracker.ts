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

import {Pose, Track} from '../types';
import {TrackerConfig} from './interfaces/config_interfaces';

/**
 * A stateful tracker for associating detections between frames. This is an
 * abstract base class that performs generic mechanics. Implementations must
 * inherit from this class.
 */
export abstract class Tracker {
  private tracks: Track[];
  private readonly max_tracks: number;
  private readonly max_age: number;

  constructor(config: TrackerConfig) {
    this.max_tracks = config.max_tracks;
    this.max_age = config.max_age;
  }

  /**
   * Tracks person instances across frames based on detections.
   * @param poses An array of detected `Pose`s.
   * @param timestamp The timestamp associated with the incoming poses.
   * @returns An updated array of `Pose`s with tracking id properties.
   */
  apply(
      poses: Pose[], timestamp: number): Pose[] {
    const sim_matrix = this.computeSimilarity(poses);
    this.assignTracks(poses, sim_matrix);
    this.removeStaleTracks(timestamp);
    return poses;
  }

  /**
   * Computes pairwise similarity scores between detections and tracks, based
   * on detected features.
   * @param poses An array of detected `Pose`s.
   * @returns A 2D array of shape [num_det, num_tracks] with pairwise similarity scores between detections and tracks.
   */
  computeSimilarity(
      poses: Pose[]): number[][] {
    const num_poses = poses.length;
    const num_tracks = this.tracks.length;
    if (!(num_poses && num_tracks)) {
      return [[]];
    }

    const sim_matrix = [];
    for (let i = 0; i < num_poses; ++i) {
      const sim_row = [];
      for (let j = 0; j < num_tracks; ++j) {
        sim_row.push(this.similarityFn(poses[i], this.tracks[j]));
      }
      sim_matrix.push(sim_row);
    }
    return sim_matrix;
  }

  /**
   * Computes the similarity between a single detection and track.
   * @param pose A single `Pose`.
   * @returns A scalar which represents the similarity between the given pose
   * detection and track.
   */
  abstract similarityFn(
      pose: Pose, track: Track): number;

  /**
   * Performs an optimization to link detections with tracks. The `poses`
   * array is updated in place by providing an `id` property. If incoming 
   * detections are not linked with existing tracks, new tracks will be created.
   * @param poses An array of detected `Pose's.
   */
  assignTracks(
      poses: Pose[], similarity_matrix: number[][]): void {
    //TODO: Implement optimization and track store mechanics.
  }


  /**
   * Removes tracks that have not been linked with poses for a length of time.
   */
  removeStaleTracks(timestamp): void {
    this.tracks = this.tracks.filter(track => {
      return timestamp - track.lastTimestamp < this.max_age;
    })
  }

  /**
   * Removes specific tracks, based on their ids.
   */
  remove(...ids): void {
    this.tracks = this.tracks.filter(track => !ids.includes(track.id));
  }

   /**
   * Resets tracks.
   */
  reset(): void {
    this.tracks = [];
  }
}
