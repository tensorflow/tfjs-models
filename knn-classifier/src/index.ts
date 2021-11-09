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
import {Tensor, Tensor1D, Tensor2D, util} from '@tensorflow/tfjs-core';
import {concatWithNulls, topK} from './util';
export {version} from './version';

/**
 * A K-nearest neighbors (KNN) classifier that allows fast
 * custom model training on top of any tensor input. Useful for transfer
 * learning with an embedding from another pretrained model.
 */
export class KNNClassifier {
  // The full concatenated dataset that is constructed lazily before making a
  // prediction.
  private trainDatasetMatrix: Tensor2D;

  // Individual class datasets used when adding examples. These get concatenated
  // into the full trainDatasetMatrix when a prediction is made.
  private classDatasetMatrices: {[label: string]: Tensor2D} = {};
  private classExampleCount: {[label: string]: number} = {};

  private exampleShape: number[];
  private labelToClassId: {[label: string]: number} = {};
  private nextClassId = 0;

  /**
   * Adds the provided example to the specified class.
   */
  addExample(example: Tensor, label: number|string): void {
    if (this.exampleShape == null) {
      this.exampleShape = example.shape;
    }
    if (!util.arraysEqual(this.exampleShape, example.shape)) {
      throw new Error(
          `Example shape provided, ${example.shape} does not match ` +
          `previously provided example shapes ${this.exampleShape}.`);
    }

    this.clearTrainDatasetMatrix();

    if (!(label in this.labelToClassId)) {
      this.labelToClassId[label] = this.nextClassId++;
    }

    tf.tidy(() => {
      const normalizedExample =
        this.normalizeVectorToUnitLength(tf.reshape(example, [example.size]));
      const exampleSize = normalizedExample.shape[0];

      if (this.classDatasetMatrices[label] == null) {
        this.classDatasetMatrices[label] =
          tf.reshape(normalizedExample, [1, exampleSize]);
      } else {
        const newTrainLogitsMatrix =
          tf.concat<tf.Tensor2D>([
            tf.reshape(this.classDatasetMatrices[label],
                          [this.classExampleCount[label], exampleSize]),
            tf.reshape(normalizedExample, [1, exampleSize])
          ], 0);

        this.classDatasetMatrices[label].dispose();
        this.classDatasetMatrices[label] = newTrainLogitsMatrix;
      }

      tf.keep(this.classDatasetMatrices[label]);

      if (this.classExampleCount[label] == null) {
        this.classExampleCount[label] = 0;
      }
      this.classExampleCount[label]++;
    });
  }

  /**
   * This method return distances between the input and all examples in the
   * dataset.
   *
   * @param input The input example.
   * @returns cosine similarities for each entry in the database.
   */
  private similarities(input: Tensor): Tensor1D {
    return tf.tidy(() => {
      const normalizedExample =
        this.normalizeVectorToUnitLength(tf.reshape(input, [input.size]));
      const exampleSize = normalizedExample.shape[0];

      // Lazily create the logits matrix for all training examples if necessary.
      if (this.trainDatasetMatrix == null) {
        let newTrainLogitsMatrix = null;

        for (const label in this.classDatasetMatrices) {
          newTrainLogitsMatrix = concatWithNulls(
              newTrainLogitsMatrix, this.classDatasetMatrices[label]);
        }
        this.trainDatasetMatrix = newTrainLogitsMatrix;
      }

      if (this.trainDatasetMatrix == null) {
        console.warn('Cannot predict without providing training examples.');
        return null;
      }

      tf.keep(this.trainDatasetMatrix);

      const numExamples = this.getNumExamples();
      return tf.reshape(
        tf.matMul(
          tf.reshape(this.trainDatasetMatrix, [numExamples, exampleSize]),
          tf.reshape(normalizedExample, [exampleSize, 1])
        ), [numExamples]);
    });
  }

  /**
   * Predicts the class of the provided input using KNN from the previously-
   * added inputs and their classes.
   *
   * @param input The input to predict the class for.
   * @returns A dict of the top class for the input and an array of confidence
   * values for all possible classes.
   */
  async predictClass(input: Tensor, k = 3): Promise<{
    label: string,
    classIndex: number,
    confidences: {[label: string]: number}
  }> {
    if (k < 1) {
      throw new Error(
          `Please provide a positive integer k value to predictClass.`);
    }
    if (this.getNumExamples() === 0) {
      throw new Error(
          `You have not added any examples to the KNN classifier. ` +
          `Please add examples before calling predictClass.`);
    }
    const knn = tf.tidy(() => tf.cast(this.similarities(input),'float32'));
    const kVal = Math.min(k, this.getNumExamples());
    const topKIndices = topK(await knn.data() as Float32Array, kVal).indices;
    knn.dispose();

    return this.calculateTopClass(topKIndices, kVal);
  }

  /**
   * Clears the saved examples from the specified class.
   */
  clearClass(label: number|string) {
    if (this.classDatasetMatrices[label] == null) {
      throw new Error(`Cannot clear invalid class ${label}`);
    }

    this.classDatasetMatrices[label].dispose();
    delete this.classDatasetMatrices[label];
    delete this.classExampleCount[label];
    this.clearTrainDatasetMatrix();
  }

  clearAllClasses() {
    for (const label in this.classDatasetMatrices) {
      this.clearClass(label);
    }
  }

  getClassExampleCount(): {[label: string]: number} {
    return this.classExampleCount;
  }

  getClassifierDataset(): {[label: string]: Tensor2D} {
    return this.classDatasetMatrices;
  }

  getNumClasses(): number {
    return Object.keys(this.classExampleCount).length;
  }

  setClassifierDataset(classDatasetMatrices: {[label: string]: Tensor2D}) {
    this.clearTrainDatasetMatrix();

    this.classDatasetMatrices = classDatasetMatrices;
    for (const label in classDatasetMatrices) {
      this.classExampleCount[label] = classDatasetMatrices[label].shape[0];
    }
  }

  /**
   * Calculates the top class in knn prediction
   * @param topKIndices The indices of closest K values.
   * @param kVal The value of k for the k-nearest neighbors algorithm.
   */
  private calculateTopClass(topKIndices: Int32Array, kVal: number) {
    let topLabel: string;
    const confidences: {[label: string]: number} = {};

    if (topKIndices == null) {
      // No class predicted
      return {
        classIndex: this.labelToClassId[topLabel],
        label: topLabel,
        confidences
      };
    }

    const classOffsets: {[label: string]: number} = {};
    let offset = 0;
    for (const label in this.classDatasetMatrices) {
      offset += this.classExampleCount[label];
      classOffsets[label] = offset;
    }
    const votesPerClass: {[label: string]: number} = {};
    for (const label in this.classDatasetMatrices) {
      votesPerClass[label] = 0;
    }
    for (let i = 0; i < topKIndices.length; i++) {
      const index = topKIndices[i];
      for (const label in this.classDatasetMatrices) {
        if (index < classOffsets[label]) {
          votesPerClass[label]++;
          break;
        }
      }
    }

    // Compute confidences.
    let topConfidence = 0;
    for (const label in this.classDatasetMatrices) {
      const probability = votesPerClass[label] / kVal;
      if (probability > topConfidence) {
        topConfidence = probability;
        topLabel = label;
      }
      confidences[label] = probability;
    }

    return {
      classIndex: this.labelToClassId[topLabel],
      label: topLabel,
      confidences
    };
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
      const sqrtSum = tf.norm(vec);

      return tf.div(vec, sqrtSum);
    });
  }

  private getNumExamples() {
    let total = 0;
    for (const label in this.classDatasetMatrices) {
      total += this.classExampleCount[label];
    }

    return total;
  }

  dispose() {
    this.clearTrainDatasetMatrix();
    for (const label in this.classDatasetMatrices) {
      this.classDatasetMatrices[label].dispose();
    }
  }
}

export function create(): KNNClassifier {
  return new KNNClassifier();
}
