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
  private readonly maxTracks: number;
  private readonly maxAge: number;

  constructor(config: TrackerConfig) {
    this.maxTracks = config.maxTracks;
    this.maxAge = config.maxAge;
  }

  /**
   * Tracks person instances across frames based on detections.
   * @param poses An array of detected `Pose`s.
   * @param timestamp The timestamp associated with the incoming poses.
   * @returns An updated array of `Pose`s with tracking id properties.
   */
  apply(
      poses: Pose[], timestamp: number): Pose[] {
    const simMatrix = this.computeSimilarity(poses);
    this.assignTracks(poses, simMatrix);
    this.updateTracks(timestamp);
    return poses;
  }

  /**
   * Computes pairwise similarity scores between detections and tracks, based
   * on detected features.
   * @param poses An array of detected `Pose`s.
   * @returns A 2D array of shape [num_det, num_tracks] with pairwise
   * similarity scores between detections and tracks.
   */
  computeSimilarity(
      poses: Pose[]): number[][] {
    const numPoses = poses.length;
    const numTracks = this.tracks.length;
    if (!(numPoses && numTracks)) {
      return [[]];
    }

    const simMatris = [];
    for (let i = 0; i < numPoses; ++i) {
      const simRow = [];
      for (let j = 0; j < numTracks; ++j) {
        simRow.push(this.similarityFn(poses[i], this.tracks[j]));
      }
      simMatris.push(simRow);
    }
    return simMatris;
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
   * @param simMatrix A 2D array of shape [num_det, num_tracks] with pairwse
   * similarity scores between detections and tracks.
   */
  assignTracks(
      poses: Pose[], simMatrix: number[][]): void {
    //TODO: Implement optimization and track store mechanics.
  }

  /**
   * Updates the stored tracks in the tracker. Specifically, the following
   * operations are applied in order:
   * 1. Tracks that have not been linked in the past `maxAge` milliseconds are
   *    removed.
   * 2. Tracks are sorted based on freshness (i.e. the most recently linked
   *    tracks are placed at the beginning of the array and the most stale are
   *    at the end).
   * 3. The tracks array is sliced to only contain `maxTracks` tracks (i.e. the
   *    most fresh tracks).
   * @param timestamp The current timestamp in milliseconds.
   */
  updateTracks(timestamp: number): void {
    this.tracks = this.tracks.filter(track => {
      return timestamp - track.lastTimestamp < this.maxAge;
    });

    // Sort tracks from most recent to most stale, and then only keep the top
    // `maxTracks` tracks.
    this.tracks.sort((ta, tb) => tb.lastTimestamp - ta.lastTimestamp);
    this.tracks = this.tracks.slice(0, this.maxTracks);
  }

  /**
   * Removes specific tracks, based on their ids.
   */
  remove(...ids: number[]): void {
    this.tracks = this.tracks.filter(track => !ids.includes(track.id));
  }

  /**
   * Resets tracks.
   */
  reset(): void {
    this.tracks = [];
  }
}
