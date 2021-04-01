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
import * as tf from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';

import {Keypoint} from '../types';

/**
 * Encapsulates a TensorFlow person keypoint model.
 */
export class KeypointModel {
  private model: tf.GraphModel;

  /**
   * Loads the model from a URL.
   * @param url URL that points to the model.json file.
   * @param fromTfHub Indicates whether the model is hosted on TF Hub.
   */
  async load(url: string, fromTfHub = false) {
    if (!fromTfHub) {
      this.model = await tf.loadGraphModel(url);
    } else {
      this.model = await tf.loadGraphModel(url, {fromTFHub: true});
    }
  }

  /**
   * Runs inference on an image using a model that is assumed to be a person
   * keypoint model that outputs 17 keypoints.
   * @param inputImage 4D tensor containing the input image. Should be of size
   *     [1, modelHeight, modelWidth, 3]. The tensor will be disposed.
   * @param executeSync Whether to execute the model synchronously.
   * @return An InferenceResult with keypoints and scores, or null if the
   *     inference call could not be executed (for example when the model was
   *     not initialized yet) or if it produced an unexpected tensor size.
   */
  async detectKeypoints(inputImage: tfc.Tensor4D, executeSync = true):
      Promise<Keypoint[]|null> {
    if (!this.model) {
      return null;
    }

    const numKeypoints = 17;

    let outputTensor;
    if (executeSync) {
      outputTensor = this.model.execute(inputImage) as tfc.Tensor;
    } else {
      outputTensor = await this.model.executeAsync(inputImage) as tfc.Tensor;
    }
    inputImage.dispose();

    // We expect an output array of shape [1, 1, 17, 3] (batch, person,
    // keypoint, coordinate + score).
    if (!outputTensor || outputTensor.shape.length !== 4 ||
        outputTensor.shape[0] !== 1 || outputTensor.shape[1] !== 1 ||
        outputTensor.shape[2] !== numKeypoints || outputTensor.shape[3] !== 3) {
      outputTensor.dispose();
      return null;
    }

    const inferenceResult = outputTensor.dataSync();
    outputTensor.dispose();

    const keypoints: Keypoint[] = [];

    for (let i = 0; i < numKeypoints; ++i) {
      keypoints[i] = {
        y: inferenceResult[i * 3],
        x: inferenceResult[i * 3 + 1],
        score: inferenceResult[i * 3 + 2]
      };
    }

    return keypoints;
  }

  dispose() {
    this.model.dispose();
  }
}
