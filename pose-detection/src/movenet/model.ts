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

/**
 * Represents the output tensor of an inference call in a type that does not
 * need to be disposed. See:
 * https://js.tensorflow.org/api/latest/#tf.Tensor.dispose
 */
interface InferenceResult {
  shape: number[];
  values: number|number[]|number[][]|number[][][]|number[][][][]|
      number[][][][][]|number[][][][][][];
}

/**
 * Encapsulates a TensorFlow model.
 */
export class Model {
  private model: tf.GraphModel;

  constructor() {}

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
   * Runs inference on an image.
   * @param inputImage 4D tensor containing the input image. Should be of size
   *     [1, modelHeight, modelWidth, 3]. The tensor will be disposed.
   * @param executeSync Whether to execute the model synchronously. A model with
   *     control flow ops can only be executed asynchronously. See:
   *     https://js.tensorflow.org/api/latest/#tf.GraphModel.executeAsync
   * @return An multidimensional array containing the values of the first output
   *     tensor or 'null' if the model was not initialized yet.
   */
  async runInference(inputImage: tfc.Tensor4D, executeSync: boolean):
      Promise<InferenceResult|null> {
    if (!this.model) {
      return null;
    }

    let outputTensor;
    if (executeSync) {
      outputTensor = this.model.execute(inputImage) as tfc.Tensor;
    } else {
      outputTensor = await this.model.executeAsync(inputImage) as tfc.Tensor;
    }
    inputImage.dispose();

    const inferenceResult = {
      'shape': outputTensor.shape,
      'values': outputTensor.arraySync(),
    } as InferenceResult;

    outputTensor.dispose();

    return inferenceResult;
  }

  dispose() {
    this.model.dispose();
  }
}
