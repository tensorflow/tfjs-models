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

import {MOVENET_CONFIG, MOVENET_ESTIMATION_CONFIG, VALID_MODELS} from './constants';
import {MoveNetEstimationConfig, MoveNetModelConfig} from './types';

export function validateModelConfig(modelConfig: MoveNetModelConfig):
    MoveNetModelConfig {
  const config = modelConfig == null ? MOVENET_CONFIG : {...modelConfig};

  if (!modelConfig.modelType) {
    modelConfig.modelType = 'SinglePose.Lightning';
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

  // Tracker will validate the trackingConfig.

  return config;
}

export function validateEstimationConfig(
    estimationConfig: MoveNetEstimationConfig): MoveNetEstimationConfig {
  const config = estimationConfig == null ? MOVENET_ESTIMATION_CONFIG :
                                            {...estimationConfig};

  return config;
}

function isObject(item: unknown) {
  return (item != null && typeof item === 'object' && !Array.isArray(item));
}

/**
 * Merges two objects that may contain nested objects. Outputs a copy, so does
 * not modify the inputs.
 *
 * Solution taken from:
 * https://stackoverflow.com/questions/27936772/how-to-deep-merge-instead-of-shallow-merge
 *
 * @param target The object to merge into.
 * @param source The object to merge from.
 * @return A copy of `target` with all the properties of `source` merged into
 * it.
 */
export function mergeDeep(
    target: {[name: string]: unknown},
    source: {[name: string]: unknown}): {[name: string]: unknown} {
  const output = Object.assign({}, target);
  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach(key => {
      if (isObject(source[key])) {
        if (!(key in target)) {
          Object.assign(output, {[key]: source[key]});
        } else {
          output[key] = mergeDeep(target[key] as {}, source[key] as {});
        }
      } else {
        Object.assign(output, {[key]: source[key]});
      }
    });
  }
  return output;
}
