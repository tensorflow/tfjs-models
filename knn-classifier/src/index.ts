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
import * as tf from '@tensorflow/tfjs';
import {Tensor, Tensor1D, Tensor2D, util} from '@tensorflow/tfjs';
import {isNumber} from 'util';

import {concatWithNulls, topK} from './util';

/**
 * A K-nearest neighbors (KNN) image classifier that allows fast
 * custom model training on top of mobilenet
 */
export class KNNClassifier {
  private trainDatasetMatrix: Tensor2D;

  private classDatasetMatrix: {[classId: number]: Tensor2D} = {};
  private classExampleCount: {[classId: number]: number} = {};

  private exampleShape: number[];

  /**
   * Clears the saved images from the specified class.
   */
  clearClass(classIndex: number) {
    if (this.classDatasetMatrix[classIndex] == null) {
      throw new Error('Cannot clear invalid class ${classIndex}');
    }

    this.classDatasetMatrix[classIndex] = null;
    delete this.classDatasetMatrix[classIndex];
    this.classExampleCount[classIndex] = 0;
    delete this.classExampleCount[classIndex];
    this.clearTrainDatasetMatrix();
  }

  /**
   * Adds the provided example to the specified class.
   */
  addExample(example: Tensor, classIndex: number): void {
    if (this.exampleShape == null) {
      this.exampleShape = example.shape;
    }
    if (!util.arraysEqual(this.exampleShape, example.shape)) {
      throw new Error(
          `Example shape provided, ${example.shape} does not match ` +
          `previously provided example shapes ${this.exampleShape}.`);
    }
    if (!isNumber(classIndex)) {
      throw new Error(`classIndex must be an integer, got ${classIndex}.`);
    }

    this.clearTrainDatasetMatrix();

    tf.tidy(() => {
      const normalizedExample =
          this.normalizeVectorToUnitLength(example.flatten());
      const exampleSize = normalizedExample.shape[0];

      if (this.classDatasetMatrix[classIndex] == null) {
        this.classDatasetMatrix[classIndex] =
            normalizedExample.as2D(1, exampleSize);
      } else {
        const newTrainLogitsMatrix =
            this.classDatasetMatrix[classIndex]
                .as2D(this.classExampleCount[classIndex], exampleSize)
                .concat(normalizedExample.as2D(1, exampleSize), 0);

        this.classDatasetMatrix[classIndex].dispose();
        this.classDatasetMatrix[classIndex] = newTrainLogitsMatrix;
      }

      tf.keep(this.classDatasetMatrix[classIndex]);

      if (this.classExampleCount[classIndex] == null) {
        this.classExampleCount[classIndex] = 0;
      }
      this.classExampleCount[classIndex]++;
    });
  }

  /**
   * This method returns the K-nearest neighbors as distances in the database.
   *
   * @param example The input example.
   * @returns cosine distances for each entry in the database.
   */
  private knn(example: Tensor): Tensor1D {
    return tf.tidy(() => {
      const normalizedExample =
          this.normalizeVectorToUnitLength(example.flatten());
      const exampleSize = normalizedExample.shape[0];

      // Lazily create the logits matrix for all training images if necessary.
      if (this.trainDatasetMatrix == null) {
        let newTrainLogitsMatrix = null;

        for (const i in this.classDatasetMatrix) {
          newTrainLogitsMatrix =
              concatWithNulls(newTrainLogitsMatrix, this.classDatasetMatrix[i]);
        }
        this.trainDatasetMatrix = newTrainLogitsMatrix;
      }

      if (this.trainDatasetMatrix == null) {
        console.warn('Cannot predict without providing training images.');
        return null;
      }

      tf.keep(this.trainDatasetMatrix);

      const numExamples = this.getNumExamples();
      return this.trainDatasetMatrix.as2D(numExamples, exampleSize)
          .matMul(normalizedExample.as2D(exampleSize, 1))
          .as1D();
    });
  }

  /**
   * Predicts the class of the provided image using KNN from the previously-
   * added images and their classes.
   *
   * @param example The image to predict the class for.
   * @returns A dict of the top class for the image and an array of confidence
   * values for all possible classes.
   */
  async predictClass(example: Tensor, k = 3):
      Promise<{classIndex: number, confidences: {[classId: number]: number}}> {
    const knn = this.knn(example).asType('float32');

    const kVal = Math.min(k, this.getNumExamples());
    const topKIndices = topK(await knn.data() as Float32Array, kVal).indices;
    knn.dispose();

    return this.calculateTopClass(topKIndices, kVal);
  }

  getClassExampleCount(): {[classId: number]: number} {
    return this.classExampleCount;
  }

  getClassLogitsMatrices(): {[classId: number]: Tensor2D} {
    return this.classDatasetMatrix;
  }

  getNumClasses(): number {
    return Object.keys(this.classExampleCount).length;
  }

  setDataset(classDatasetMatrix: {[classId: number]: Tensor2D}) {
    this.clearTrainDatasetMatrix();

    this.classDatasetMatrix = classDatasetMatrix;
    for (const i in classDatasetMatrix) {
      this.classExampleCount[i] = classDatasetMatrix[i].shape[0];
    }
  }

  /**
   * Calculates the top class in knn prediction
   */
  private calculateTopClass(topKIndices: Int32Array, kVal: number) {
    let imageClass = -1;
    const confidences: {[classId: number]: number} = {};

    if (topKIndices == null) {
      // No class predicted
      return {classIndex: imageClass, confidences};
    }

    const indicesForClasses = [];
    const topKCountsForClasses = [];
    for (const i in this.classDatasetMatrix) {
      topKCountsForClasses.push(0);
      let num = this.classExampleCount[i];
      if (+i > 0) {
        num += indicesForClasses[+i - 1];
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
    for (const i in this.classDatasetMatrix) {
      const probability = topKCountsForClasses[i] / kVal;
      if (probability > topConfidence) {
        topConfidence = probability;
        imageClass = +i;
      }
      confidences[i] = probability;
    }

    return {classIndex: imageClass, confidences};
  }

  /**
   * Clear the lazily-loaded train logits matrix due to a change in
   * training data.
   */
  private clearTrainDatasetMatrix() {
    if (this.trainDatasetMatrix != null) {
      this.trainDatasetMatrix.dispose();
      this.trainDatasetMatrix = null;
    }
  }

  /**
   * Normalize the provided vector to unit length.
   */
  private normalizeVectorToUnitLength(vec: Tensor1D) {
    return tf.tidy(() => {
      const sqrtSum = vec.norm();

      return tf.div(vec, sqrtSum);
    });
  }

  private getNumExamples() {
    let total = 0;
    for (const i in this.classDatasetMatrix) {
      total += this.classExampleCount[+i];
    }

    return total;
  }

  dispose() {
    this.clearTrainDatasetMatrix();
    for (const i in this.classDatasetMatrix) {
      this.classDatasetMatrix[i].dispose();
    }
  }
}

export function create(): KNNClassifier {
  return new KNNClassifier();
}
