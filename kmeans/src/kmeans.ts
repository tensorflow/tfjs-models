import {Tensor} from '@tensorflow/tfjs-core';
import {kMeansMain, initCentroids, assignToNearest} from './training';

export interface KMeansArgs {
  nClusters: number;
  maxIter?: number;
  tol?: number;
}

export class KMeansClustering {
  // A public property that can be set by Callbacks to order early stopping
  // during `fit()` calls.
  protected stopTraining_: boolean;
  protected isTraining: boolean;
  private inputs: Tensor;
  private outputs: Tensor;
  private clusterCenters: Tensor;
  private readonly nClusters: number;
  private readonly maxIter: number;
  private readonly tol: number;

  constructor({nClusters = 8, maxIter = 300, tol = 0.0001}) {
    this.isTraining = false;
    this.nClusters = nClusters;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(x: Tensor): Tensor {
    if (this.isTraining) {
      throw new Error(
        'Cannot start training because another fit() call is ongoing.'
      );
    }
    if (!(x instanceof Tensor)) {
      throw new Error('Input must be tensor');
    }
    this.inputs = x;

    this.init();
    this.fitLoop();

    this.isTraining = false;
    return this.outputs;
  }

  protected init(): void {
    this.clusterCenters = initCentroids(this.inputs, this.nClusters);
  }

  protected fitLoop(): void {
    const outs = kMeansMain(this.inputs, this.nClusters, this.maxIter, this.tol);
    this.outputs = outs;
  }

  predict(x: Tensor): Tensor {
    if (this.isTraining) {
      throw new Error('Cannot start prediction because fit() call is ongoing.');
    }
    const inputs = x;
    return assignToNearest(inputs, this.clusterCenters);
  }

  fitPredict(x: Tensor): Tensor {
    return this.fit(x);
  }
}

export function kMeans(args: KMeansArgs): KMeansClustering {
  return new KMeansClustering(args);
}
