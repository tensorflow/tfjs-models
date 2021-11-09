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

import {TrackerConfig} from '../calculators/interfaces/config_interfaces';
import {TrackerType} from '../calculators/types';

import {DEFAULT_BOUNDING_BOX_TRACKER_CONFIG, DEFAULT_KEYPOINT_TRACKER_CONFIG, MOVENET_CONFIG, MOVENET_ESTIMATION_CONFIG, MULTIPOSE_LIGHTNING, VALID_MODELS} from './constants';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

export function validateModelConfig(modelConfig: MoveNetModelConfig):
    MoveNetModelConfig {
  const config = modelConfig == null ? MOVENET_CONFIG : {...modelConfig};

  if (config.modelType == null) {
    config.modelType = 'SinglePose.Lightning';
  } else if (VALID_MODELS.indexOf(config.modelType) < 0) {
    throw new Error(
        `Invalid architecture ${config.modelType}. ` +
        `Should be one of ${VALID_MODELS}`);
  }

  if (config.enableSmoothing == null) {
    config.enableSmoothing = true;
  }

  if (config.minPoseScore != null &&
      (config.minPoseScore < 0.0 || config.minPoseScore > 1.0)) {
    throw new Error(`minPoseScore should be between 0.0 and 1.0`);
  }

  if (config.multiPoseMaxDimension != null &&
      (config.multiPoseMaxDimension % 32 !== 0 ||
       config.multiPoseMaxDimension < 32)) {
    throw new Error(
        `multiPoseMaxDimension must be a multiple of 32 and higher than 0`);
  }

  if (config.modelType === MULTIPOSE_LIGHTNING &&
      config.enableTracking == null) {
    config.enableTracking = true;
  }

  if (config.modelType === MULTIPOSE_LIGHTNING &&
      config.enableTracking === true) {
    if (config.trackerType == null) {
      config.trackerType = TrackerType.BoundingBox;
    }
    if (config.trackerType === TrackerType.Keypoint) {
      if (config.trackerConfig != null) {
        config.trackerConfig = mergeKeypointTrackerConfig(config.trackerConfig);
      } else {
        config.trackerConfig = DEFAULT_KEYPOINT_TRACKER_CONFIG;
      }
    } else if (config.trackerType === TrackerType.BoundingBox) {
      if (config.trackerConfig != null) {
        config.trackerConfig =
            mergeBoundingBoxTrackerConfig(config.trackerConfig);
      } else {
        config.trackerConfig = DEFAULT_BOUNDING_BOX_TRACKER_CONFIG;
      }
    } else {
      throw new Error('Tracker type not supported by MoveNet');
    }

    // We don't need to validate the trackerConfig here because the tracker will
    // take care of that.
  }

  return config;
}

export function validateEstimationConfig(
    estimationConfig: MoveNetEstimationConfig): MoveNetEstimationConfig {
  const config = estimationConfig == null ? MOVENET_ESTIMATION_CONFIG :
                                            {...estimationConfig};

  return config;
}

function mergeBaseTrackerConfig(
    defaultConfig: TrackerConfig, userConfig: TrackerConfig): TrackerConfig {
  const mergedConfig: TrackerConfig = {
    maxTracks: defaultConfig.maxTracks,
    maxAge: defaultConfig.maxAge,
    minSimilarity: defaultConfig.minSimilarity,
  };

  if (userConfig.maxTracks != null) {
    mergedConfig.maxTracks = userConfig.maxTracks;
  }
  if (userConfig.maxAge != null) {
    mergedConfig.maxAge = userConfig.maxAge;
  }
  if (userConfig.minSimilarity != null) {
    mergedConfig.minSimilarity = userConfig.minSimilarity;
  }

  return mergedConfig;
}

export function mergeKeypointTrackerConfig(userConfig: TrackerConfig):
    TrackerConfig {
  const mergedConfig =
      mergeBaseTrackerConfig(DEFAULT_KEYPOINT_TRACKER_CONFIG, userConfig);

  mergedConfig.keypointTrackerParams = {
      ...DEFAULT_KEYPOINT_TRACKER_CONFIG.keypointTrackerParams};

  if (userConfig.keypointTrackerParams != null) {
    if (userConfig.keypointTrackerParams.keypointConfidenceThreshold != null) {
      mergedConfig.keypointTrackerParams.keypointConfidenceThreshold =
          userConfig.keypointTrackerParams.keypointConfidenceThreshold;
    }
    if (userConfig.keypointTrackerParams.keypointFalloff != null) {
      mergedConfig.keypointTrackerParams.keypointFalloff =
          userConfig.keypointTrackerParams.keypointFalloff;
    }
    if (userConfig.keypointTrackerParams.minNumberOfKeypoints != null) {
      mergedConfig.keypointTrackerParams.minNumberOfKeypoints =
          userConfig.keypointTrackerParams.minNumberOfKeypoints;
    }
  }

  return mergedConfig;
}

export function mergeBoundingBoxTrackerConfig(userConfig: TrackerConfig):
    TrackerConfig {
  const mergedConfig =
      mergeBaseTrackerConfig(DEFAULT_BOUNDING_BOX_TRACKER_CONFIG, userConfig);

  return mergedConfig;
}
