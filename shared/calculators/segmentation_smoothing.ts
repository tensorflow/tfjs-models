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
import {GPGPUProgram, MathBackendWebGL} from '@tensorflow/tfjs-backend-webgl';
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
  if (tf.getBackend() === 'webgl') {
    // Same as implementation in the else case but reduces number of shader
    // calls to 1 instead of 17.
    return smoothSegmentationWebGL(prevMask, newMask, config);
  }
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

function smoothSegmentationWebGL(
    prevMask: tf.Tensor2D, newMask: tf.Tensor2D,
    config: SegmentationSmoothingConfig): tf.Tensor2D {
  const ratio = config.combineWithPreviousRatio.toFixed(2);
  const program: GPGPUProgram = {
    variableNames: ['prevMask', 'newMask'],
    outputShape: prevMask.shape,
    userCode: `
  void main() {
      ivec2 coords = getOutputCoords();
      int height = coords[0];
      int width = coords[1];

      float prevMaskValue = getPrevMask(height, width);
      float newMaskValue = getNewMask(height, width);

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
      const float c1 = 5.68842;
      const float c2 = -0.748699;
      const float c3 = -57.8051;
      const float c4 = 291.309;
      const float c5 = -624.717;
      float t = newMaskValue - 0.5;
      float x = t * t;

      float uncertainty =
        1.0 - min(1.0, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

      float outputValue = newMaskValue + (prevMaskValue - newMaskValue) *
                             (uncertainty * ${ratio});

      setOutput(outputValue);
    }
`
  };
  const webglBackend = tf.backend() as MathBackendWebGL;

  return tf.tidy(() => {
    const outputTensorInfo =
        webglBackend.compileAndRun(program, [prevMask, newMask]);
    return tf.engine().makeTensorFromDataId(
               outputTensorInfo.dataId, outputTensorInfo.shape,
               outputTensorInfo.dtype) as tf.Tensor2D;
  });
}
