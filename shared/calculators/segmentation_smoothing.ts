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
import * as tf from '@tensorflow/tfjs-core';

import {SegmentationSmoothingConfig} from './interfaces/config_interfaces';

/**
 * A calculator for mixing two segmentation masks together, based on an
 * uncertantity probability estimate.
 * @param prevMaks Segmentation mask from previous image.
 * @param newMask Segmentation mask of current image.
 * @param config Contains ratio of amount of previous mask to blend with
 *     current.
 *
 * @returns Image mask.
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/image/segmentation_smoothing_calculator.cc
export function smoothSegmentation(
    prevMask: tf.Tensor2D, newMask: tf.Tensor2D,
    config: SegmentationSmoothingConfig): tf.Tensor2D {
  return tf.tidy(() => {
    /*
     * Assume p := newMaskValue
     * H(p) := 1 + (p * log(p) + (1-p) * log(1-p)) / log(2)
     * uncertainty alpha(p) =
     *   Clamp(1 - (1 - H(p)) * (1 - H(p)), 0, 1) [squaring the
     * uncertainty]
     *
     * The following polynomial approximates uncertainty alpha as a
     * function of (p + 0.5):
     */
    const c1 = 5.68842;
    const c2 = -0.748699;
    const c3 = -57.8051;
    const c4 = 291.309;
    const c5 = -624.717;
    const t = tf.sub(newMask, 0.5);
    const x = tf.square(t);

    // Per element calculation is: 1.0 - Math.min(1.0, x * (c1 + x * (c2 + x
    // * (c3 + x * (c4 + x * c5))))).

    const uncertainty = tf.sub(
        1,
        tf.minimum(
            1,
            tf.mul(
                x,
                tf.add(
                    c1,
                    tf.mul(
                        x,
                        tf.add(
                            c2,
                            tf.mul(
                                x,
                                tf.add(
                                    c3,
                                    tf.mul(
                                        x, tf.add(c4, tf.mul(x, c5)))))))))));

    // Per element calculation is: newMaskValue + (prevMaskValue -
    // newMaskValue) * (uncertainty * combineWithPreviousRatio).
    return tf.add(
        newMask,
        tf.mul(
            tf.sub(prevMask, newMask),
            tf.mul(uncertainty, config.combineWithPreviousRatio)));
  });
}
