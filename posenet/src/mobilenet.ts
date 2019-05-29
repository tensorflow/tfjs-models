import * as tf from '@tensorflow/tfjs';

import {OutputStride} from '.';
import {BaseModel, PoseNetResolution} from './posenet_model';

export class MobileNet implements BaseModel {
  readonly model: tf.GraphModel

  readonly outputStride: OutputStride
  readonly inputResolution: PoseNetResolution;

  private PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
  private ONE = tf.scalar(1.0);

  constructor(
      model: tf.GraphModel, inputResolution: PoseNetResolution,
      outputStride: OutputStride) {
    this.model = model;
    const inputShape =
        this.model.inputs[0].shape as [number, number, number, number];
    [inputShape[1], inputShape[2]];
    this.inputResolution = inputResolution;
    this.outputStride = outputStride;
  }

  predict(input: tf.Tensor3D): {[key: string]: tf.Tensor3D} {
    return tf.tidy(() => {
      const normalized = tf.div(input.toFloat(), this.PREPROCESS_DIVISOR);

      const preprocessedInput = tf.sub(normalized, this.ONE) as tf.Tensor3D;

      const asBatch = preprocessedInput.expandDims(0);

      const result = this.model.predict(asBatch) as tf.Tensor<tf.Rank>[];
      const [offsets4d, displacementFwd4d, displacementBwd4d, heatmaps4d] =
          result;

      const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze() as tf.Tensor3D;
      const displacementFwd = displacementFwd4d.squeeze() as tf.Tensor3D;
      const displacementBwd = displacementBwd4d.squeeze() as tf.Tensor3D;

      return {
        heatmapScores, offsets: offsets as tf.Tensor3D,
            displacementFwd: displacementFwd as tf.Tensor3D,
            displacementBwd: displacementBwd as tf.Tensor3D
      }
    });
  }

  dispose() {
    this.model.dispose();
  }
}
