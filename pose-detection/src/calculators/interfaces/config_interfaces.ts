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

export interface TrackerConfig {
  maxTracks: number;  // The maximum number of tracks that an internal tracker
                      // will maintain. Note that this number should be set
                      // larger than EstimationConfig.maxPoses. How to set this
                      // number requires experimentation with a given detector,
                      // but a good starting place is about 3 * maxPoses.
  maxAge: number;     // The maximum duration of time (in milliseconds) that a
                   // track can exist without being linked with a new detection
                   // before it is removed. Set this value large if you would
                   // like to recover people that are not detected for long
                   // stretches of time (at the cost of potential false
                   // re-identifications).
  minSimilarity: number;  // New poses will only be linked with tracks if the
                          // similarity score exceeds this threshold.
  keypointTrackerParams?: KeypointTrackerConfig;  // Keypoint tracker params.
  boundingBoxTrackerParams?: BoundingBoxTrackerConfig;  // Box tracker params.
}
// A tracker that links detections (i.e. poses) and tracks based on keypoint
// similarity.
export interface KeypointTrackerConfig {
  keypointConfidenceThreshold: number;  // The minimum keypoint confidence
                                        // threshold. A keypoint is only
                                        // compared in the OKS calculation if
                                        // both the new detected keypoint and
                                        // the corresponding track keypoint have
                                        // confidences above this threshold.

  keypointFalloff: number[];     // Per-keypoint falloff in OKS calculation.
  minNumberOfKeypoints: number;  // The minimum number of keypoints that are
                                 // necessary for computing OKS. If the number
                                 // of confident keypoints (between a pose and
                                 // track) are under this value, an OKS of 0.0
                                 // will be given.
}
// A tracker that links detections (i.e. poses) and tracks based on bounding
// box similarity.
export interface BoundingBoxTrackerConfig {}
