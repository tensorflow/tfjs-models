/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as tfl from '@tensorflow/tfjs-layers';

const getExactlyOneTensor = (xs: tf.Tensor[]): tf.Tensor => {
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new Error(`Expected Tensor length to be 1; got ${xs.length}.`);
    }
    return xs[0];
  }
  return xs;
};

class ChannelPadding extends tfl.layers.Layer {
  private padding: number;

  constructor(config: {padding: number}) {
    super({});

    this.padding = config.padding;
  }

  computeOutputShape(inputShape: [number, number, number, number]):
      [number, number, number, number] {
    const [batch, dim1, dim2, values] = inputShape;
    return [batch, dim1, dim2, values + this.padding];
  }

  call(inputs: tf.Tensor4D[]): tf.Tensor4D {
    const input = getExactlyOneTensor(inputs);
    return tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, this.padding]], 0.0) as
        tf.Tensor4D;
  }

  static get className() {
    return 'ChannelPadding';
  }
}

tf.serialization.registerClass(ChannelPadding);
