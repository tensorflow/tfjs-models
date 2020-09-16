/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

export class ModelWeights {
  private variables: {[varName: string]: tf.Tensor};

  constructor(variables: {[varName: string]: tf.Tensor}) {
    this.variables = variables;
  }

  weights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/weights`] as tf.Tensor4D;
  }

  depthwiseBias(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/biases`] as tf.Tensor1D;
  }

  convBias(layerName: string) {
    return this.depthwiseBias(layerName);
  }

  depthwiseWeights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/depthwise_weights`] as
        tf.Tensor4D;
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
