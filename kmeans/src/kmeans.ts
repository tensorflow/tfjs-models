import * as tf from '@tensorflow/tfjs-core';
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

  protected async init(): Promise<void> {
    tf.dispose(this.clusterCenters);
    this.clusterCenters = await initCentroids(this.inputs, this.nClusters);
  }

  protected fitSingle(): void {
    const fitOutput = kMeansFitOneCycle(
      this.inputs,
      this.clusterCenters,
      this.nClusters,
      this.tol
    );
    tf.dispose(this.outputs);
    tf.dispose(this.clusterCenters);
    this.outputs = fitOutput.nearest;
    this.clusterCenters = fitOutput.centroids;
  }

  async fit(x: Tensor): Promise<void> {
    // console.log(this.clusterCenters.dataSync());
    if (this.isTraining) {
      throw new Error(
        'Cannot start training because another fit() call is ongoing.'
      );
    }
    if (!(x instanceof Tensor)) {
      throw new Error('Input must be tensor');
    }
    this.inputs = x.toFloat();
    await this.init();
    for (let i = 0; i < this.maxIter; i++) {
      this.fitSingle();
    }
    this.isTraining = false;
    tf.dispose(this.inputs);
  }

  async fitOneCycle(x: Tensor): Promise<Int32Array> {
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
      await this.init();
    }
    this.fitSingle();
    this.isTraining = false;
    tf.dispose(this.inputs);

    const res = (await this.outputs.data()) as Int32Array;
    tf.dispose(this.outputs);
    return res;
  }

  async predict(x: Tensor): Promise<Int32Array> {
    if (this.isTraining) {
      throw new Error('Cannot start prediction because fit() call is ongoing.');
    }
    const outputs = assignToNearest(x, this.clusterCenters);
    const res = (await outputs.data()) as Int32Array;
    tf.dispose(outputs);
    return res;
  }

  async fitPredict(x: Tensor): Promise<Int32Array> {
    await this.fit(x);
    const res = (await this.outputs.data()) as Int32Array;
    tf.dispose(this.outputs);
    return res;
  }
}

export function kMeans(args: KMeansArgs): KMeansClustering {
  return new KMeansClustering(args);
}
