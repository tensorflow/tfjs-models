import * as tfc from '@tensorflow/tfjs-core'
import {Tensor, Tensor1D, Tensor2D, Scalar, tidy, util} from '@tensorflow/tfjs';
import {kMeans, assignToNearest} from './util';

// ALL_DISTANCES
// CLUSTER_CENTERS_VAR_NAME
// CLUSTER_INDEX
// COSINE_DISTANCE
// KMEANS_PLUS_PLUS_INIT
// RANDOM_INIT
// SCORE
// SQUARED_EUCLIDEAN_DISTANCE

// summary
// init
// cluster_centers
// evaluate // evaluate based on some metric
// fit
// transform
// predict

/**
 * A K-nearest neighbors (KNN) classifier that allows fast
 * custom model training on top of any tensor input. Useful for transfer
 * learning with an embedding from another pretrained model.
 */
export class KMeansClusteringModel implements tfc.InferenceModel {
  // // The full concatenated dataset that is constructed lazily before making a
  // // prediction.
  // private trainDatasetMatrix: Tensor2D;

  // A public property that can be set by Callbacks to order early stopping
  // during `fit()` calls.
  private stopTraining_: boolean;
  private isTraining: boolean;

  metrics: string[]|{[outputName: string]: string};
  metricsNames: string[];

  private nClusters: number;
  private init: string;
  private nInit: number;
  private maxIter: number;
  private tol: number;
  private precomputeDistances: string;
  private algorithm: string;
  clusterCenters: Tensor;

  constructor({nClusters = 8, init='k-means++', nInit=10, maxIter=300, tol=0.0001, precomputeDistances='auto', algorithm='auto'}) {
    this.isTraining = false;
    this.nClusters = nClusters;
    this.init = init;
    this.nInit = nInit;
    this.maxIter = maxIter;
    this.tol = tol;
    this.precomputeDistances = precomputeDistances;
    this.algorithm = algorithm;
  }

  _init(k: number): void {
    this.nClusters = k;
  }

  // todo: read about how compile is implemented in Model
  evaluate(
      x: Tensor|Tensor[], y: Tensor|Tensor[]): Scalar|Scalar[] {

  }

  // why fit is async while predict / evaluate are not
  async fit(
      x: Tensor|Tensor[], y: Tensor|Tensor[]): Promise<History> {
    this.clusterCenters = kMeans(X, this.nClusters, this.maxIter, this.tol);
    return
  }

  predict(X: number[][]): Tensor2D {
    return assignToNearest(X, this.clusterCenters);
  }

  dummy(): string {
    return this.algorithm;
  }

  // /**
  //  * Adds the provided example to the specified class.
  //  */
  // addExample(example: Tensor, classIndex: number): void {
  //   if (this.exampleShape == null) {
  //     this.exampleShape = example.shape;
  //   }
  //   if (!util.arraysEqual(this.exampleShape, example.shape)) {
  //     throw new Error(
  //         `Example shape provided, ${example.shape} does not match ` +
  //         `previously provided example shapes ${this.exampleShape}.`);
  //   }
  //   if (!Number.isInteger(classIndex)) {
  //     throw new Error(`classIndex must be an integer, got ${classIndex}.`);
  //   }
  //
  //   this.clearTrainDatasetMatrix();
  //
  //   tf.tidy(() => {
  //     const normalizedExample =
  //         this.normalizeVectorToUnitLength(example.flatten());
  //     const exampleSize = normalizedExample.shape[0];
  //
  //     if (this.classDatasetMatrices[classIndex] == null) {
  //       this.classDatasetMatrices[classIndex] =
  //           normalizedExample.as2D(1, exampleSize);
  //     } else {
  //       const newTrainLogitsMatrix =
  //           this.classDatasetMatrices[classIndex]
  //               .as2D(this.classExampleCount[classIndex], exampleSize)
  //               .concat(normalizedExample.as2D(1, exampleSize), 0);
  //
  //       this.classDatasetMatrices[classIndex].dispose();
  //       this.classDatasetMatrices[classIndex] = newTrainLogitsMatrix;
  //     }
  //
  //     tf.keep(this.classDatasetMatrices[classIndex]);
  //
  //     if (this.classExampleCount[classIndex] == null) {
  //       this.classExampleCount[classIndex] = 0;
  //     }
  //     this.classExampleCount[classIndex]++;
  //   });
  // }
  //
  // /**
  //  * This method return distances between the input and all examples in the
  //  * dataset.
  //  *
  //  * @param input The input example.
  //  * @returns cosine similarities for each entry in the database.
  //  */
  // private similarities(input: Tensor): Tensor1D {
  //   return tf.tidy(() => {
  //     const normalizedExample =
  //         this.normalizeVectorToUnitLength(input.flatten());
  //     const exampleSize = normalizedExample.shape[0];
  //
  //     // Lazily create the logits matrix for all training examples if necessary.
  //     if (this.trainDatasetMatrix == null) {
  //       let newTrainLogitsMatrix = null;
  //
  //       for (const i in this.classDatasetMatrices) {
  //         newTrainLogitsMatrix = concatWithNulls(
  //             newTrainLogitsMatrix, this.classDatasetMatrices[i]);
  //       }
  //       this.trainDatasetMatrix = newTrainLogitsMatrix;
  //     }
  //
  //     if (this.trainDatasetMatrix == null) {
  //       console.warn('Cannot predict without providing training examples.');
  //       return null;
  //     }
  //
  //     tf.keep(this.trainDatasetMatrix);
  //
  //     const numExamples = this.getNumExamples();
  //     return this.trainDatasetMatrix.as2D(numExamples, exampleSize)
  //         .matMul(normalizedExample.as2D(exampleSize, 1))
  //         .as1D();
  //   });
  // }
  //
  // /**
  //  * Predicts the class of the provided input using KNN from the previously-
  //  * added inputs and their classes.
  //  *
  //  * @param input The input to predict the class for.
  //  * @returns A dict of the top class for the input and an array of confidence
  //  * values for all possible classes.
  //  */
  // async predictClass(input: Tensor, k = 3):
  //     Promise<{classIndex: number, confidences: {[classId: number]: number}}> {
  //   if (k < 1) {
  //     throw new Error(
  //         `Please provide a positive integer k value to predictClass.`);
  //   }
  //   if (this.getNumExamples() === 0) {
  //     throw new Error(
  //         `You have not added any exaples to the KNN classifier. ` +
  //         `Please add examples before calling predictClass.`);
  //   }
  //   const knn = tf.tidy(() => this.similarities(input).asType('float32'));
  //
  //   const kVal = Math.min(k, this.getNumExamples());
  //   const topKIndices = topK(await knn.data() as Float32Array, kVal).indices;
  //   knn.dispose();
  //
  //   return this.calculateTopClass(topKIndices, kVal);
  // }
  //
  // /**
  //  * Clears the saved examples from the specified class.
  //  */
  // clearClass(classIndex: number) {
  //   if (this.classDatasetMatrices[classIndex] == null) {
  //     throw new Error('Cannot clear invalid class ${classIndex}');
  //   }
  //
  //   delete this.classDatasetMatrices[classIndex];
  //   delete this.classExampleCount[classIndex];
  //   this.clearTrainDatasetMatrix();
  // }
  //
  // clearAllClasses() {
  //   for (const i in this.classDatasetMatrices) {
  //     this.clearClass(+i);
  //   }
  // }
  //
  // getClassExampleCount(): {[classId: number]: number} {
  //   return this.classExampleCount;
  // }
  //
  // getClassifierDataset(): {[classId: number]: Tensor2D} {
  //   return this.classDatasetMatrices;
  // }
  //
  // getNumClasses(): number {
  //   return Object.keys(this.classExampleCount).length;
  // }
  //
  // setClassifierDataset(classDatasetMatrices: {[classId: number]: Tensor2D}) {
  //   this.clearTrainDatasetMatrix();
  //
  //   this.classDatasetMatrices = classDatasetMatrices;
  //   for (const i in classDatasetMatrices) {
  //     this.classExampleCount[i] = classDatasetMatrices[i].shape[0];
  //   }
  // }
  //
  // /**
  //  * Calculates the top class in knn prediction
  //  * @param topKIndices The indices of closest K values.
  //  * @param kVal The value of k for the k-nearest neighbors algorithm.
  //  */
  // private calculateTopClass(topKIndices: Int32Array, kVal: number) {
  //   let exampleClass = -1;
  //   const confidences: {[classId: number]: number} = {};
  //
  //   if (topKIndices == null) {
  //     // No class predicted
  //     return {classIndex: exampleClass, confidences};
  //   }
  //
  //   const indicesForClasses = [];
  //   for (const i in this.classDatasetMatrices) {
  //     let num = this.classExampleCount[i];
  //     if (+i > 0) {
  //       num += indicesForClasses[+i - 1];
  //     }
  //     indicesForClasses.push(num);
  //   }
  //
  //   const topKCountsForClasses =
  //       Array(Object.keys(this.classDatasetMatrices).length).fill(0);
  //   for (let i = 0; i < topKIndices.length; i++) {
  //     for (let classId = 0; classId < indicesForClasses.length; classId++) {
  //       if (topKIndices[i] < indicesForClasses[classId]) {
  //         topKCountsForClasses[classId]++;
  //         break;
  //       }
  //     }
  //   }
  //
  //   // Compute confidences.
  //   let topConfidence = 0;
  //   for (const i in this.classDatasetMatrices) {
  //     const probability = topKCountsForClasses[i] / kVal;
  //     if (probability > topConfidence) {
  //       topConfidence = probability;
  //       exampleClass = +i;
  //     }
  //     confidences[i] = probability;
  //   }
  //
  //   return {classIndex: exampleClass, confidences};
  // }
  //
  // /**
  //  * Clear the lazily-loaded train logits matrix due to a change in
  //  * training data.
  //  */
  // private clearTrainDatasetMatrix() {
  //   if (this.trainDatasetMatrix != null) {
  //     this.trainDatasetMatrix.dispose();
  //     this.trainDatasetMatrix = null;
  //   }
  // }
  //
  // /**
  //  * Normalize the provided vector to unit length.
  //  */
  // private normalizeVectorToUnitLength(vec: Tensor1D) {
  //   return tf.tidy(() => {
  //     const sqrtSum = vec.norm();
  //
  //     return tf.div(vec, sqrtSum);
  //   });
  // }
  //
  // private getNumExamples() {
  //   let total = 0;
  //   for (const i in this.classDatasetMatrices) {
  //     total += this.classExampleCount[+i];
  //   }
  //
  //   return total;
  // }
  //
  // dispose() {
  //   this.clearTrainDatasetMatrix();
  //   for (const i in this.classDatasetMatrices) {
  //     this.classDatasetMatrices[i].dispose();
  //   }
  // }
}

export function create(): KMeansClustering {
  return new KMeansClustering({});
}
