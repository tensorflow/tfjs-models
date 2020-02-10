import {Tensor} from '@tensorflow/tfjs-core';
import {kMeansFitOneCycle, initCentroids, assignToNearest} from './training';

export interface KMeansArgs {
  nClusters?: number;
  maxIter?: number;
  tol?: number;
}

export class KMeansClustering {
  protected isTraining: boolean;
  private inputs: Tensor;
  private outputs: Tensor;
  public clusterCenters: Tensor;
  private readonly nClusters: number;
  private readonly maxIter: number;
  private readonly tol: number;

  constructor({nClusters = 8, maxIter = 300, tol = 0.0001}) {
    this.isTraining = false;
    this.nClusters = nClusters;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  protected init(): void {
    this.clusterCenters = initCentroids(this.inputs, this.nClusters);
  }

  protected fitSingle(): void {
    const fitOutput = kMeansFitOneCycle(
      this.inputs,
      this.clusterCenters,
      this.nClusters,
      this.tol
    );
    this.outputs = fitOutput.nearest;
    this.clusterCenters = fitOutput.centroids;
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
    this.inputs = x.toFloat();
    this.init();
    for (let i = 0; i < this.maxIter; i++) {
      this.fitSingle();
    }
    this.isTraining = false;
    return this.outputs;
  }

  fitOneCycle(x: Tensor): Tensor {
    if (this.isTraining) {
      throw new Error(
        'Cannot start training because another fit() call is ongoing.'
      );
    }
    if (!(x instanceof Tensor)) {
      throw new Error('Input must be tensor');
    }
    this.inputs = x.toFloat();
    if (!this.clusterCenters) {
      this.init();
    }
    this.fitSingle();
    this.isTraining = false;
    return this.outputs;
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
