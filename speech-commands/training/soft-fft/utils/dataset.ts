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

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class Dataset {
  xs: tf.Tensor[];
  ys: tf.Tensor;
  constructor(public numClasses: number) {}

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example.
   *    It can be an image, an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be an number.
   */
  addExample(example: tf.Tensor|tf.Tensor[], label: number) {
    example = Array.isArray(example) ? example : [example];
    // One-hot encode the label.
    const y =
        tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // Dataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = example.map(tensor => tf.keep(tensor));
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = example.map(
          (tensor, index) => tf.keep(this.xs[index].concat(tensor, 0)));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.forEach(tensor => tensor.dispose());
      oldY.dispose();
      y.dispose();
    }
  }
}
