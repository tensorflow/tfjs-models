/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import {Tensor1D, Tensor2D, Tensor3D} from '@tensorflow/tfjs';
import {concatWithNulls, topK} from './util';

/**
 * A K-nearest neighbors (KNN) image classifier that allows fast
 * custom model training on top of mobilenet
 */
class KNNClassifier {
  private net: mobilenet.MobileNet;

  private trainLogitsMatrix: Tensor2D;

  private classLogitsMatrices: Tensor2D[] = [];
  private classExampleCount: number[] = [];

  private squashLogitsDenominator = tf.scalar(300);

  constructor(
      private numClasses: number, private k: number, net: mobilenet.MobileNet) {
    for (let i = 0; i < this.numClasses; i++) {
      this.classLogitsMatrices.push(null);
      this.classExampleCount.push(0);
    }

    this.net = net;
  }

  /**
   * Clears the saved images from the specified class.
   */
  clearClass(classIndex: number) {
    if (classIndex >= this.numClasses) {
      throw new Error('Cannot clear invalid class ${classIndex}');
    }

    this.classLogitsMatrices[classIndex] = null;
    this.classExampleCount[classIndex] = 0;
    this.clearTrainLogitsMatrix();
  }

  /**
   * Adds the provided image to the specified class.
   */
  addImage(image: Tensor3D, classIndex: number): void {
    if (classIndex >= this.numClasses) {
      throw new Error('Cannot add to invalid class ${classIndex}');
    }
    this.clearTrainLogitsMatrix();

    tf.tidy(() => {
      // Add the mobilenet logits for the image to the appropriate class
      // logits matrix.
      const logits = this.inferNormalizedLogits(image);
      const logitsSize = logits.shape[0];

      if (this.classLogitsMatrices[classIndex] == null) {
        this.classLogitsMatrices[classIndex] = logits.as2D(1, logitsSize);
      } else {
        const newTrainLogitsMatrix =
            this.classLogitsMatrices[classIndex]
                .as2D(this.classExampleCount[classIndex], logitsSize)
                .concat(logits.as2D(1, logitsSize), 0);

        this.classLogitsMatrices[classIndex].dispose();
        this.classLogitsMatrices[classIndex] = newTrainLogitsMatrix;
      }

      tf.keep(this.classLogitsMatrices[classIndex]);

      this.classExampleCount[classIndex]++;
    });
  }

  /**
   * This method returns the K-nearest neighbors as distances in the database.
   *
   * This unfortunately deviates from standard behavior for nearest neighbors
   * classifiers, making this method relatively useless:
   * http://scikit-learn.org/stable/modules/neighbors.html
   *
   * TODO(nsthorat): Return the class indices once we have GPU kernels for topK
   * and take. This method is useless on its own, but matches our Model API.
   *
   * @param image The input image.
   * @returns cosine distances for each entry in the database.
   */
  calculateKNNDistances(image: Tensor3D): Tensor1D {
    return tf.tidy(() => {
      const logits = this.inferNormalizedLogits(image);
      const logitsSize = logits.shape[0];

      // Lazily create the logits matrix for all training images if necessary.
      if (this.trainLogitsMatrix == null) {
        let newTrainLogitsMatrix = null;

        for (let i = 0; i < this.numClasses; i++) {
          newTrainLogitsMatrix = concatWithNulls(
              newTrainLogitsMatrix, this.classLogitsMatrices[i]);
        }
        this.trainLogitsMatrix = newTrainLogitsMatrix;
      }

      if (this.trainLogitsMatrix == null) {
        console.warn('Cannot predict without providing training images.');
        return null;
      }

      tf.keep(this.trainLogitsMatrix);

      const numExamples = this.getNumExamples();
      return this.trainLogitsMatrix.as2D(numExamples, logitsSize)
          .matMul(logits.as2D(logitsSize, 1))
          .as1D();
    });
  }

  /**
   * Predicts the class of the provided image using KNN from the previously-
   * added images and their classes.
   *
   * @param image The image to predict the class for.
   * @returns A dict of the top class for the image and an array of confidence
   * values for all possible classes.
   */
  async predictClass(image: Tensor3D):
      Promise<{classIndex: number, confidences: number[]}> {
    const knn = this.calculateKNNDistances(image).asType('float32');

    const kVal = Math.min(this.k, this.getNumExamples());
    const topKIndices = topK(await knn.data() as Float32Array, kVal).indices;
    knn.dispose();

    return this.calculateTopClass(topKIndices, kVal);
  }

  getClassExampleCount(): number[] {
    return this.classExampleCount;
  }

  getClassLogitsMatrices(): Tensor2D[] {
    return this.classLogitsMatrices;
  }

  setClassLogitsMatrices(classLogitsMatrices: Tensor2D[]) {
    this.classLogitsMatrices = classLogitsMatrices;
    this.classExampleCount = classLogitsMatrices.map(
        (tensor: Tensor2D) => tensor != null ? tensor.shape[0] : 0);
    this.clearTrainLogitsMatrix();
  }

  /**
   * Calculates the top class in knn prediction
   */
  private calculateTopClass(topKIndices: Int32Array, kVal: number) {
    let imageClass = -1;
    const confidences = new Array<number>(this.numClasses);

    if (topKIndices == null) {
      // No class predicted
      return {classIndex: imageClass, confidences};
    }

    const indicesForClasses = [];
    const topKCountsForClasses = [];
    for (let i = 0; i < this.numClasses; i++) {
      topKCountsForClasses.push(0);
      let num = this.classExampleCount[i];
      if (i > 0) {
        num += indicesForClasses[i - 1];
      }
      indicesForClasses.push(num);
    }

    for (let i = 0; i < topKIndices.length; i++) {
      for (let classForEntry = 0; classForEntry < indicesForClasses.length;
           classForEntry++) {
        if (topKIndices[i] < indicesForClasses[classForEntry]) {
          topKCountsForClasses[classForEntry]++;
          break;
        }
      }
    }

    let topConfidence = 0;
    for (let i = 0; i < this.numClasses; i++) {
      const probability = topKCountsForClasses[i] / kVal;
      if (probability > topConfidence) {
        topConfidence = probability;
        imageClass = i;
      }
      confidences[i] = probability;
    }

    return {classIndex: imageClass, confidences};
  }

  /**
   * Calculates normalized logits of image
   * @param image image to pass through mobilenet
   */
  private inferNormalizedLogits(image: tf.Tensor<tf.Rank.R3>) {
    const logits = this.net.infer(image).as1D();
    return this.normalizeVector(logits);
  }

  /**
   * Clear the lazily-loaded train logits matrix due to a change in
   * training data.
   */
  private clearTrainLogitsMatrix() {
    if (this.trainLogitsMatrix != null) {
      this.trainLogitsMatrix.dispose();
      this.trainLogitsMatrix = null;
    }
  }

  /**
   * Normalize the provided vector to unit length.
   */
  private normalizeVector(vec: Tensor1D) {
    // This hack is here for numerical stability on devices without floating
    // point textures. We divide by a constant so that the sum doesn't overflow
    // our fixed point precision. Remove this once we use floating point
    // intermediates with proper dynamic range quantization.

    const squashedVec = tf.div(vec, this.squashLogitsDenominator);
    const sqrtSum = squashedVec.square().sum().sqrt();

    return tf.div(squashedVec, sqrtSum);
  }

  private getNumExamples() {
    let total = 0;
    for (let i = 0; i < this.classExampleCount.length; i++) {
      total += this.classExampleCount[i];
    }

    return total;
  }

  dispose() {
    this.clearTrainLogitsMatrix();
    this.classLogitsMatrices.forEach(
        classLogitsMatrix => classLogitsMatrix.dispose());
    this.squashLogitsDenominator.dispose();
  }
}

/**
 * Load KNN Model.
 * This will load mobilenet, and prepare the classifier for a
 * predifined number of classes
 *
 * @param numClasses number of classes to predict
 * @param topK number of neighbors to compare with during prediction
 */
async function load(numClasses: number, topK: number) {
  const model = await mobilenet.load();
  return new KNNClassifier(numClasses, topK, model);
}

export {KNNClassifier, load};
