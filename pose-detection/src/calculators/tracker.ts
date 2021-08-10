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

import {Track} from './interfaces/common_interfaces';
import {TrackerConfig} from './interfaces/config_interfaces';
import {validateTrackerConfig} from './tracker_utils';

/**
 * A stateful tracker for associating detections between frames. This is an
 * abstract base class that performs generic mechanics. Implementations must
 * inherit from this class.
 */
export abstract class Tracker {
  protected tracks: Track[];
  private readonly maxTracks: number;
  private readonly maxAge: number;
  private readonly minSimilarity: number;
  private nextID: number;

  constructor(config: TrackerConfig) {
    validateTrackerConfig(config);
    this.tracks = [];
    this.maxTracks = config.maxTracks;
    this.maxAge = config.maxAge * 1000;  // Convert msec to usec.
    this.minSimilarity = config.minSimilarity;
    this.nextID = 1;
  }

  /**
   * Tracks person instances across frames based on detections.
   * @param poses An array of detected `Pose`s.
   * @param timestamp The timestamp associated with the incoming poses, in
   * microseconds.
   * @returns An updated array of `Pose`s with tracking id properties.
   */
  apply(poses: Pose[], timestamp: number): Pose[] {
    this.filterOldTracks(timestamp);
    const simMatrix = this.computeSimilarity(poses);
    this.assignTracks(poses, simMatrix, timestamp);
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
  abstract computeSimilarity(poses: Pose[]): number[][];

  /**
   * Returns a copy of the stored tracks.
   */
  getTracks(): Track[] {
    return this.tracks.slice();
  }

  /**
   * Returns a Set of active track IDs.
   */
  getTrackIDs(): Set<number> {
    return new Set(this.tracks.map(track => track.id));
  }

  /**
   * Filters tracks based on their age.
   * @param timestamp The current timestamp in microseconds.
   */
  filterOldTracks(timestamp: number): void {
    this.tracks = this.tracks.filter(track => {
      return timestamp - track.lastTimestamp <= this.maxAge;
    });
  }

  /**
   * Performs a greedy optimization to link detections with tracks. The `poses`
   * array is updated in place by providing an `id` property. If incoming
   * detections are not linked with existing tracks, new tracks will be created.
   * @param poses An array of detected `Pose`s. It's assumed that poses are
   * sorted from most confident to least confident.
   * @param simMatrix A 2D array of shape [num_det, num_tracks] with pairwise
   * similarity scores between detections and tracks.
   * @param timestamp The current timestamp in microseconds.
   */
  assignTracks(poses: Pose[], simMatrix: number[][], timestamp: number): void {
    const unmatchedTrackIndices = Array.from(Array(simMatrix[0].length).keys());
    const detectionIndices = Array.from(Array(poses.length).keys());
    const unmatchedDetectionIndices: number[] = [];

    for (const detectionIndex of detectionIndices) {
      if (unmatchedTrackIndices.length === 0) {
        unmatchedDetectionIndices.push(detectionIndex);
        continue;
      }

      // Assign the detection to the track which produces the highest pairwise
      // similarity score, assuming the score exceeds the minimum similarity
      // threshold.
      let maxTrackIndex = -1;
      let maxSimilarity = -1;
      for (const trackIndex of unmatchedTrackIndices) {
        const similarity = simMatrix[detectionIndex][trackIndex];
        if (similarity >= this.minSimilarity && similarity > maxSimilarity) {
          maxTrackIndex = trackIndex;
          maxSimilarity = similarity;
        }
      }
      if (maxTrackIndex >= 0) {
        // Link the detection with the highest scoring track.
        let linkedTrack = this.tracks[maxTrackIndex];
        linkedTrack = Object.assign(
            linkedTrack,
            this.createTrack(poses[detectionIndex], timestamp, linkedTrack.id));
        poses[detectionIndex].id = linkedTrack.id;
        const index = unmatchedTrackIndices.indexOf(maxTrackIndex);
        unmatchedTrackIndices.splice(index, 1);
      } else {
        unmatchedDetectionIndices.push(detectionIndex);
      }
    }

    // Spawn new tracks for all unmatched detections.
    for (const detectionIndex of unmatchedDetectionIndices) {
      const newTrack = this.createTrack(poses[detectionIndex], timestamp);
      this.tracks.push(newTrack);
      poses[detectionIndex].id = newTrack.id;
    }
  }

  /**
   * Updates the stored tracks in the tracker. Specifically, the following
   * operations are applied in order:
   * 1. Tracks are sorted based on freshness (i.e. the most recently linked
   *    tracks are placed at the beginning of the array and the most stale are
   *    at the end).
   * 2. The tracks array is sliced to only contain `maxTracks` tracks (i.e. the
   *    most fresh tracks).
   * @param timestamp The current timestamp in microseconds.
   */
  updateTracks(timestamp: number): void {
    // Sort tracks from most recent to most stale, and then only keep the top
    // `maxTracks` tracks.
    this.tracks.sort((ta, tb) => tb.lastTimestamp - ta.lastTimestamp);
    this.tracks = this.tracks.slice(0, this.maxTracks);
  }

  /**
   * Creates a track from information in a pose.
   * @param pose A `Pose`.
   * @param timestamp The current timestamp in microseconds.
   * @param trackID The id to assign to the new track. If not provided,
   * will assign the next available id.
   * @returns A `Track`.
   */
  createTrack(pose: Pose, timestamp: number, trackID?: number): Track {
    const track: Track = {
      id: trackID || this.nextTrackID(),
      lastTimestamp: timestamp,
      keypoints: [...pose.keypoints].map(keypoint => ({...keypoint}))
    };
    if (pose.box !== undefined) {
      track.box = {...pose.box};
    }
    return track;
  }

  /**
   * Returns the next free track ID.
   */
  nextTrackID() {
    const nextID = this.nextID;
    this.nextID += 1;
    return nextID;
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
