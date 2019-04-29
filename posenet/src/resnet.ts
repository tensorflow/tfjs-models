import * as tf from '@tensorflow/tfjs';
import {BackboneInterface, PoseNetResolution} from './posenet_model';
import { OutputStride } from '.';

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32')
      input = input.toFloat();
    const ImageNetMean = tf.tensor([-123.15, -115.90, -103.06]);
    return input.add(ImageNetMean);
  })
}

export default class ResNet implements BackboneInterface {
  readonly model: tf.GraphModel
  readonly outputStride: number
  readonly inputDimensions: [number, number]

  SUPPORTED_RESOLUTION: Array<PoseNetResolution> = [257, 513];

  constructor(model: tf.GraphModel, outputStride: number) {
    this.model = model;
    this.outputStride = outputStride;
    const inputShape = this.model.inputs[0].shape as [number, number, number, number];
    this.inputDimensions = [inputShape[1], inputShape[2]];
  }

  predict(input: tf.Tensor3D, outputStride: OutputStride = 32): {[key: string]: tf.Tensor3D} {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [displacementFwd4d, displacementBwd4d, offsets4d, heatmaps4d] =
          this.model.predict(asBatch) as tf.Tensor<tf.Rank>[];

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
