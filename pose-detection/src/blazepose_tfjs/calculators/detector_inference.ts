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
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {splitDetectionResult} from './split_detection_result';

export type DetectorInferenceResult = {
  boxes: tf.Tensor2D,
  scores: tf.Tensor1D
};

export function detectorInference(
    imageTensor: tf.Tensor4D,
    poseDetectorModel: tfconv.GraphModel): DetectorInferenceResult {
  return tf.tidy(() => {
    const detectionResult =
        poseDetectorModel.predict(imageTensor) as tf.Tensor3D;
    const [scores, rawBoxes] = splitDetectionResult(detectionResult);
    // Shape [896, 12]
    const rawBoxes2d = tf.squeeze(rawBoxes);
    // Shape [896]
    const scores1d = tf.squeeze(scores);

    return {boxes: rawBoxes2d as tf.Tensor2D, scores: scores1d as tf.Tensor1D};
  });
}
